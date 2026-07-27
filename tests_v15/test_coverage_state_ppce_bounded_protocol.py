from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import cure_lite.experiment.coverage_state_ppce_bounded_runner as runner
from cure_lite.coverage_state_level_set import CoverageStateLevelSetConfig
from cure_lite.coverage_state_phase_preserving import (
    CURELitePhasePreservingCoverageStateLevelSet,
    CoverageStatePhasePreservingConfig,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_SEED,
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_dataset_free import (
    run_coverage_state_phase_preserving_dataset_free_gate,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)


@pytest.fixture(scope="module")
def ppce_authorization():
    population = build_coverage_state_bounded_population(
        make_bounded_training_scalar_cache()
    )
    preflight = prepare_coverage_state_bounded_preflight(population)
    dataset_free = (
        run_coverage_state_phase_preserving_dataset_free_gate()
    )
    return runner.prepare_coverage_state_ppce_bounded_run_authorization(
        preflight,
        dataset_free,
    )


def test_ppce_authorization_binds_model_suite_receipt_and_parent(
    ppce_authorization,
) -> None:
    authorization = ppce_authorization
    assert authorization.training_authorized
    assert authorization.objective_suite == (
        "support_oriented_response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    expected_config = runner.expected_coverage_state_ppce_config(
        authorization.preflight
    )
    assert authorization.expected_parameter_count == (
        expected_config.expected_parameter_count
    )
    assert authorization.dataset_free_receipt_fingerprint == (
        authorization.dataset_free_receipt.receipt_fingerprint
    )
    assert authorization.parent_v15b_complete_fingerprint == (
        "13cc94f4f5140031fc050ac8d1726e13f9e5e1bbfa8a433bda28783088121f95"
    )
    assert authorization.parent_v15b_complete_sha256 == (
        "58460fde25d08123231e2ab1ae5767f46ae3e40896b605b9e77c144413f6a896"
    )
    assert authorization.parent_v15b_source_manifest_sha256 == (
        "d5d5df197eab3bf4423777a4192f7d1bc0781518a54d9e56d37a8dbb48d9da8f"
    )
    assert authorization.parent_v15b_source_archive_sha256 == (
        "e6ced21bef5926cb4fd6b9c79181980614eef3bf0fd7c14ac1cead63815cc069"
    )
    payload = authorization.canonical_payload()
    assert payload["runtime_splits"] == ["D_R"]
    assert payload["formal_training_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False


def test_ppce_authorization_rejects_parent_or_structure_drift(
    ppce_authorization,
) -> None:
    with pytest.raises(ValueError, match="binding changed"):
        replace(
            ppce_authorization,
            parent_v15b_complete_fingerprint="0" * 64,
        )
    with pytest.raises(ValueError, match="binding changed"):
        replace(
            ppce_authorization,
            expected_parameter_count=1,
        )
    with pytest.raises(ValueError, match="binding changed"):
        replace(
            ppce_authorization,
            dataset_free_receipt_fingerprint="0" * 64,
        )
    with pytest.raises(PermissionError, match="model config"):
        ppce_authorization.verify_model_config(
            CoverageStatePhasePreservingConfig(
                feature_channels=64,
                feature_stride=4,
                width=31,
            )
        )
    with pytest.raises(PermissionError, match="model config"):
        ppce_authorization.verify_model_config(
            CoverageStateLevelSetConfig(
                feature_channels=64,
                feature_stride=4,
                width=32,
            )
        )


def test_ppce_result_checks_require_three_exact_ppce_models(
    ppce_authorization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = ppce_authorization
    config = runner.expected_coverage_state_ppce_config(
        authorization.preflight
    )
    models = tuple(
        (
            name,
            CURELitePhasePreservingCoverageStateLevelSet(config),
        )
        for name in authorization.objective_suite
    )
    training = SimpleNamespace(
        results=tuple(
            SimpleNamespace(objective=name)
            for name in authorization.objective_suite
        ),
        models=models,
    )
    diagnostics = (
        (
            "support_oriented_response_joint",
            SimpleNamespace(
                bounded_gate_passed=True,
                config=SimpleNamespace(
                    input_representation="phase_preserving"
                ),
            ),
        ),
        (
            "identity_joint",
            SimpleNamespace(
                bounded_gate_passed=False,
                config=SimpleNamespace(
                    input_representation="phase_preserving"
                ),
            ),
        ),
        (
            "separable_endpoint",
            SimpleNamespace(
                bounded_gate_passed=False,
                config=SimpleNamespace(
                    input_representation="phase_preserving"
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_bounded_result_checks",
        lambda *args, **kwargs: (
            ("execution_and_fairness", True),
            ("authorized_model_config", False),
            ("zero_level_gates", False),
        ),
    )
    checks = dict(
        runner._ppce_bounded_result_checks(
            authorization,
            training,
            diagnostics,
        )
    )
    assert checks["candidate_original_zero_level_gates"] is True
    assert checks["control_diagnostics_complete"] is True
    assert checks["all_models_exact_ppce_class"] is True
    assert checks["all_models_same_ppce_config"] is True
    assert checks["all_models_expected_parameter_count"] is True
    assert checks["authorized_model_config"] is True
    assert all(checks.values())

    changed_models = (
        models[0],
        (
            "identity_joint",
            runner.CURELiteCoverageStateLevelSet(
                CoverageStateLevelSetConfig(
                    feature_channels=64,
                    feature_stride=4,
                    width=32,
                )
            ),
        ),
        models[2],
    )
    changed_training = SimpleNamespace(
        results=training.results,
        models=changed_models,
    )
    changed = dict(
        runner._ppce_bounded_result_checks(
            authorization,
            changed_training,
            diagnostics,
        )
    )
    assert changed["all_models_exact_ppce_class"] is False
    assert changed["authorized_model_config"] is False


def test_ppce_runner_dispatches_fixed_suite_without_real_training(
    ppce_authorization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = ppce_authorization
    config = runner.expected_coverage_state_ppce_config(
        authorization.preflight
    )
    models = tuple(
        (
            name,
            CURELitePhasePreservingCoverageStateLevelSet(config).train(),
        )
        for name in authorization.objective_suite
    )
    training = SimpleNamespace(
        results=tuple(
            SimpleNamespace(objective=name)
            for name in authorization.objective_suite
        ),
        models=models,
        verify_unchanged=lambda: None,
        canonical_payload=lambda: {"training": "stub"},
    )
    observed: dict[str, object] = {}

    def train(*args, **kwargs):
        observed["model_config"] = args[0]
        observed["training_config"] = kwargs["config"]
        observed["authorization"] = kwargs["authorization"]
        return training

    monkeypatch.setattr(
        runner,
        (
            "train_matched_coverage_state_"
            "phase_preserving_support_oriented_objectives"
        ),
        train,
    )

    def evaluate(model, cache, *, device, config):
        assert type(model) is (
            CURELitePhasePreservingCoverageStateLevelSet
        )
        assert model.training is False
        assert config.input_representation == "phase_preserving"
        return SimpleNamespace(
            bounded_gate_passed=True,
            canonical_payload=lambda: {"bounded_gate_passed": True},
        )

    monkeypatch.setattr(
        runner,
        "evaluate_coverage_state_zero_level_checkpoint",
        evaluate,
    )
    monkeypatch.setattr(
        runner,
        "_ppce_bounded_result_checks",
        lambda *args, **kwargs: (("orchestration_stub", True),),
    )
    result = (
        runner.run_coverage_state_ppce_support_oriented_bounded_400(
            authorization,
            config,
            device="cpu",
        )
    )
    assert observed["model_config"] is config
    assert observed["authorization"] is authorization
    assert observed["training_config"].seed == (
        COVERAGE_STATE_BOUNDED_SEED
    )
    payload = result.canonical_payload()
    assert payload["expected_parameter_count"] == (
        config.expected_parameter_count
    )
    assert payload["runtime_splits"] == ["D_R"]
    assert payload["formal_training_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False
