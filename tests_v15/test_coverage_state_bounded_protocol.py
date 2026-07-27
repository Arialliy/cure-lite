from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
)
import cure_lite.experiment.coverage_state_bounded_runner as bounded_runner
from cure_lite.experiment.coverage_state_bounded_runner import (
    COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
    COVERAGE_STATE_BOUNDED_MODEL_WIDTH,
    CoverageStateCompletionRootedBoundedRunAuthorization,
    prepare_coverage_state_completion_rooted_bounded_run_authorization,
    prepare_coverage_state_bounded_run_authorization,
    run_coverage_state_bounded_400,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_dataset_free import (
    recompute_coverage_state_dataset_free_checks,
    run_coverage_state_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
)
from cure_lite.train.coverage_state_fused_step import (
    COVERAGE_STATE_LEGACY_MATCHED_OBJECTIVES,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)


def test_bounded_population_is_exact_replay_and_sixteen_per_role() -> None:
    first = build_coverage_state_bounded_population(
        make_bounded_training_scalar_cache()
    )
    replay = build_coverage_state_bounded_population(
        make_bounded_training_scalar_cache()
    )
    assert first.canonical_payload() == replay.canonical_payload()
    assert first.population_fingerprint == replay.population_fingerprint
    assert len(first.factual_miss_record_ids) == (
        COVERAGE_STATE_BOUNDED_ROLE_COUNT
    )
    assert len(first.factual_no_miss_record_ids) == (
        COVERAGE_STATE_BOUNDED_ROLE_COUNT
    )
    assert len(first.clean_positive_pair_ids) == (
        COVERAGE_STATE_BOUNDED_ROLE_COUNT
    )
    assert len(first.component_null_pair_ids) == (
        COVERAGE_STATE_BOUNDED_ROLE_COUNT
    )
    assert len(first.identity_null_pair_ids) == (
        COVERAGE_STATE_BOUNDED_ROLE_COUNT
    )
    assert len(first.scalar_hidden_diagnostic_pair_ids) == 1
    assert len(first.cache.natural_records) == 32
    assert len(first.cache.clean_positive_records) == 16
    assert len(first.cache.component_null_records) == 16
    assert len(first.cache.diagnostic_pair_records) == 17
    assert first.canonical_payload()["D_V_accessed"] is False
    assert first.canonical_payload()["D_T_accessed"] is False


def test_bounded_population_rejects_nonprotocol_seed() -> None:
    with pytest.raises(ValueError, match="fixes seed=42"):
        build_coverage_state_bounded_population(
            make_bounded_training_scalar_cache(),
            seed=43,
        )


def test_bounded_population_detects_source_tensor_mutation() -> None:
    bounded = build_coverage_state_bounded_population(
        make_bounded_training_scalar_cache()
    )
    bounded.source_cache.raw_catalog.natural_records[0].feature[0, 0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="changed"):
        bounded.verify_unchanged()


def test_bounded_preflight_closes_schedule_and_exposure_gate() -> None:
    population = build_coverage_state_bounded_population(
        make_bounded_training_scalar_cache()
    )
    preflight = prepare_coverage_state_bounded_preflight(population)
    assert preflight.training_authorized is True
    assert preflight.failed_checks == ()
    assert preflight.schedule.config.updates == 400
    assert preflight.canonical_payload()["formal_training_authorized"] is False
    preflight.verify_unchanged()


@pytest.fixture(scope="module")
def bounded_authorization_inputs():
    population = build_coverage_state_bounded_population(
        make_bounded_training_scalar_cache()
    )
    preflight = prepare_coverage_state_bounded_preflight(population)
    dataset_free = run_coverage_state_dataset_free_gate()
    return preflight, dataset_free


def test_bounded_authorization_binds_dataset_free_and_real_preflight(
    bounded_authorization_inputs,
) -> None:
    preflight, dataset_free = bounded_authorization_inputs
    authorization = prepare_coverage_state_bounded_run_authorization(
        preflight,
        dataset_free,
    )
    assert authorization.training_authorized is True
    assert len(authorization.authorization_fingerprint) == 64
    assert len(authorization.implementation_fingerprint) == 64
    assert tuple(
        path for path, _ in authorization.implementation_binding
    ) == COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS
    assert all(
        len(digest) == 64
        for _, digest in authorization.implementation_binding
    )
    assert len(authorization.model_config_fingerprint) == 64
    authorization.verify_for_run(
        cache=preflight.population.cache,
        schedule=preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
    )
    feature_channels = int(
        preflight.population.cache.raw_catalog.natural_records[
            0
        ].feature.shape[1]
    )
    authorization.verify_model_config(
        CoverageStateLevelSetConfig(
            feature_channels=feature_channels,
            feature_stride=(
                preflight.population.cache.raw_catalog.feature_stride
            ),
            width=COVERAGE_STATE_BOUNDED_MODEL_WIDTH,
        )
    )


def test_bounded_authorization_rejects_model_or_implementation_drift(
    bounded_authorization_inputs,
) -> None:
    preflight, dataset_free = bounded_authorization_inputs
    authorization = prepare_coverage_state_bounded_run_authorization(
        preflight,
        dataset_free,
    )
    first = preflight.population.cache.raw_catalog.natural_records[0]
    with pytest.raises(PermissionError, match="model config"):
        authorization.verify_model_config(
            CoverageStateLevelSetConfig(
                feature_channels=int(first.feature.shape[1]),
                feature_stride=(
                    preflight.population.cache.raw_catalog.feature_stride
                ),
                width=COVERAGE_STATE_BOUNDED_MODEL_WIDTH + 1,
            )
        )
    with pytest.raises(ValueError, match="binding changed"):
        replace(
            authorization,
            implementation_fingerprint="0" * 64,
        )


def test_completion_rooted_authorization_binds_candidate_and_parent(
    bounded_authorization_inputs,
) -> None:
    preflight, dataset_free = bounded_authorization_inputs
    authorization = (
        prepare_coverage_state_completion_rooted_bounded_run_authorization(
            preflight,
            dataset_free,
        )
    )
    assert isinstance(
        authorization,
        CoverageStateCompletionRootedBoundedRunAuthorization,
    )
    assert authorization.training_authorized
    assert authorization.objective_suite == (
        "completion_rooted_response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    payload = authorization.canonical_payload()
    assert payload["candidate_objective"] == (
        "completion_rooted_response_joint"
    )
    assert payload["parent_v15_complete_fingerprint"] == (
        "faaa2395623f5edfa0e56ab849d20305b73df1e7b3446b22b834279a2637d14b"
    )
    with pytest.raises(ValueError, match="binding changed"):
        replace(
            authorization,
            candidate_objective="response_joint",
        )


def test_completion_rooted_gate_qualifies_candidate_not_controls(
    bounded_authorization_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, dataset_free = bounded_authorization_inputs
    authorization = (
        prepare_coverage_state_completion_rooted_bounded_run_authorization(
            preflight,
            dataset_free,
        )
    )
    training = SimpleNamespace(
        results=tuple(
            SimpleNamespace(objective=name)
            for name in authorization.objective_suite
        )
    )
    diagnostics = (
        (
            "completion_rooted_response_joint",
            SimpleNamespace(bounded_gate_passed=True),
        ),
        (
            "identity_joint",
            SimpleNamespace(bounded_gate_passed=False),
        ),
        (
            "separable_endpoint",
            SimpleNamespace(bounded_gate_passed=False),
        ),
    )
    monkeypatch.setattr(
        bounded_runner,
        "_bounded_result_checks",
        lambda *args, **kwargs: (
            ("generic_execution_and_fairness", True),
            ("zero_level_gates", False),
        ),
    )
    checks = dict(
        bounded_runner._completion_rooted_bounded_result_checks(
            authorization,
            training,
            diagnostics,
        )
    )
    assert "zero_level_gates" not in checks
    assert checks["candidate_zero_level_gates"] is True
    assert checks["control_diagnostics_complete"] is True
    assert all(checks.values())

    failed_candidate = (
        (
            "completion_rooted_response_joint",
            SimpleNamespace(bounded_gate_passed=False),
        ),
        *diagnostics[1:],
    )
    failed_checks = dict(
        bounded_runner._completion_rooted_bounded_result_checks(
            authorization,
            training,
            failed_candidate,
        )
    )
    assert failed_checks["candidate_zero_level_gates"] is False


def test_failed_dataset_free_receipt_cannot_authorize_bounded_training(
    bounded_authorization_inputs,
) -> None:
    preflight, dataset_free = bounded_authorization_inputs
    failed_training_results = (
        replace(dataset_free.training_results[0], losses_finite=False),
        *dataset_free.training_results[1:],
    )
    failed = replace(
        dataset_free,
        training_results=failed_training_results,
        checks=recompute_coverage_state_dataset_free_checks(
            dataset_free.case_results,
            failed_training_results,
            dataset_free.completion_root_probes,
        ),
    )
    authorization = prepare_coverage_state_bounded_run_authorization(
        preflight,
        failed,
    )
    assert authorization.training_authorized is False
    with pytest.raises(PermissionError, match="does not permit"):
        authorization.verify_for_run(
            cache=preflight.population.cache,
            schedule=preflight.schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
        )


def test_deterministic_execution_restores_cpu_rng_and_backend_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1807)
    expected_state = torch.get_rng_state().clone()
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CPU execution must not read a CUDA RNG")
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CPU execution must not write a CUDA RNG")
        ),
    )
    with bounded_runner._deterministic_execution("cpu"):
        torch.manual_seed(99)
        _ = torch.rand(8)
        torch.backends.cudnn.benchmark = False
    assert torch.equal(torch.get_rng_state(), expected_state)
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
    ) == previous


def test_bounded_runner_switches_trained_models_to_eval_before_diagnostics(
    bounded_authorization_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, dataset_free = bounded_authorization_inputs
    authorization = prepare_coverage_state_bounded_run_authorization(
        preflight,
        dataset_free,
    )
    first = preflight.population.cache.raw_catalog.natural_records[0]
    config = CoverageStateLevelSetConfig(
        feature_channels=int(first.feature.shape[1]),
        feature_stride=(
            preflight.population.cache.raw_catalog.feature_stride
        ),
        width=COVERAGE_STATE_BOUNDED_MODEL_WIDTH,
    )
    models = tuple(
        (
            objective.value,
            CURELiteCoverageStateLevelSet(config).train(),
        )
        for objective in COVERAGE_STATE_LEGACY_MATCHED_OBJECTIVES
    )
    training = SimpleNamespace(
        results=tuple(
            SimpleNamespace(objective=name) for name, _ in models
        ),
        models=models,
        verify_unchanged=lambda: None,
    )
    diagnostic = SimpleNamespace()

    monkeypatch.setattr(
        bounded_runner,
        "train_matched_coverage_state_objectives",
        lambda *args, **kwargs: training,
    )

    def evaluate(model, cache, *, device):
        assert model.training is False
        assert cache is preflight.population.cache
        assert torch.device(device) == torch.device("cpu")
        return diagnostic

    monkeypatch.setattr(
        bounded_runner,
        "evaluate_coverage_state_zero_level_checkpoint",
        evaluate,
    )
    monkeypatch.setattr(
        bounded_runner,
        "_bounded_result_checks",
        lambda *args, **kwargs: (("orchestration_stub", True),),
    )
    result = run_coverage_state_bounded_400(
        authorization,
        config,
        device="cpu",
    )
    assert result.bounded_gate_passed
    assert all(not model.training for _, model in result.training.models)
