from __future__ import annotations

from copy import deepcopy
from inspect import signature

import pytest
import torch

import cure_lite.experiment.coverage_state_training as training_module
from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.coverage_state_phase_preserving import (
    CURELitePhasePreservingCoverageStateLevelSet,
    CoverageStatePhasePreservingConfig,
)
from cure_lite.experiment.coverage_state_training import (
    CoverageStateMatchedTrainingConfig,
    coverage_state_model_fingerprint,
    train_matched_coverage_state_completion_rooted_objectives,
    train_matched_coverage_state_objectives,
    train_matched_coverage_state_phase_preserving_support_oriented_objectives,
    train_matched_coverage_state_support_oriented_objectives,
    train_coverage_state_objective,
)
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
)
from tests_v15.coverage_state_test_helpers import TOY_STRIDE
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


def _model() -> CURELiteCoverageStateLevelSet:
    return CURELiteCoverageStateLevelSet(
        CoverageStateLevelSetConfig(
            feature_channels=2,
            feature_stride=TOY_STRIDE,
            width=4,
        )
    )


def _run(objective: CoverageStatePairObjective):
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=2,
            steps_per_epoch=2,
        ),
    )
    torch.manual_seed(101)
    model = _model()
    initial_state = deepcopy(model.state_dict())
    initial = coverage_state_model_fingerprint(model)
    result = train_coverage_state_objective(
        model,
        torch.optim.SGD(model.parameters(), lr=1.0e-2),
        cache,
        schedule,
        objective=objective,
        device="cpu",
        expected_initial_model_fingerprint=initial,
    )
    return result, initial_state


@pytest.mark.parametrize(
    "objective",
    tuple(CoverageStatePairObjective),
)
def test_training_runner_closes_exact_compute_ledger(
    objective: CoverageStatePairObjective,
) -> None:
    result, _ = _run(objective)
    assert result.objective == objective.value
    assert result.completed_updates == 4
    assert result.forward_calls == 4
    assert result.backward_calls == 4
    assert result.optimizer_steps == 4
    assert result.logical_state_evaluations == 48
    assert result.finite_state_audits == 5
    assert len(result.device_cache_fingerprint) == 64
    assert result.device_cache_resident_bytes > 0
    assert result.execution_device == "cpu"
    assert len(result.optimizer_config_fingerprint) == 64
    assert len(result.epoch_logs) == 2
    assert result.initial_model_fingerprint != result.final_model_fingerprint
    latency = dict(result.first_nonzero_gradient_update)
    assert latency["phase_projection.weight"] == 0
    assert latency["phase_projection.bias"] == 0
    assert latency["input_projection.weight"] <= 2
    assert latency["spatial_mixing.weight"] <= 2


def test_training_runner_is_exactly_reproducible_on_cpu() -> None:
    first, first_initial = _run(
        CoverageStatePairObjective.RESPONSE_JOINT
    )
    replay, replay_initial = _run(
        CoverageStatePairObjective.RESPONSE_JOINT
    )
    assert all(
        torch.equal(value, replay_initial[name])
        for name, value in first_initial.items()
    )
    assert first.canonical_payload() == replay.canonical_payload()
    assert first.result_fingerprint == replay.result_fingerprint


def test_training_runner_rejects_initial_state_mismatch_before_updates() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=2,
        ),
    )
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="frozen initial state"):
        train_coverage_state_objective(
            model,
            optimizer,
            cache,
            schedule,
            objective=CoverageStatePairObjective.IDENTITY_JOINT,
            device="cpu",
            expected_initial_model_fingerprint="0" * 64,
        )
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )
    assert not optimizer.state


def test_training_runner_rejects_nonempty_optimizer_before_updates() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=2,
        ),
    )
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    optimizer.state[next(iter(model.parameters()))]["step"] = torch.tensor(
        1.0
    )
    before = deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="fresh empty optimizer"):
        train_coverage_state_objective(
            model,
            optimizer,
            cache,
            schedule,
            objective=CoverageStatePairObjective.RESPONSE_JOINT,
            device="cpu",
            expected_initial_model_fingerprint=(
                coverage_state_model_fingerprint(model)
            ),
        )
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )


@pytest.mark.parametrize(
    "config",
    (
        CoverageStateScheduleConfig(
            seed=42,
            epochs=10,
            steps_per_epoch=40,
        ),
        CoverageStateScheduleConfig(
            seed=43,
            epochs=10,
            steps_per_epoch=40,
        ),
        CoverageStateScheduleConfig.formal(seed=42),
    ),
)
def test_protected_training_scope_requires_explicit_authorization(
    config: CoverageStateScheduleConfig,
) -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(cache, config)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
    before = deepcopy(model.state_dict())
    with pytest.raises(PermissionError, match="explicit prerequisite-bound"):
        train_coverage_state_objective(
            model,
            optimizer,
            cache,
            schedule,
            objective=CoverageStatePairObjective.RESPONSE_JOINT,
            device="cpu",
            expected_initial_model_fingerprint=(
                coverage_state_model_fingerprint(model)
            ),
        )
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )
    assert not optimizer.state


def test_public_single_objective_api_exposes_no_verified_fast_path() -> None:
    parameters = signature(train_coverage_state_objective).parameters
    assert "_cache_already_verified" not in parameters
    assert "_schedule_already_verified" not in parameters
    assert "_authorization_already_verified" not in parameters
    assert "_device_cache" not in parameters
    assert "_device_cache_already_verified" not in parameters
    assert "_defer_device_cache_content_verification" not in parameters


def test_protected_scope_rejects_structural_noop_authorization() -> None:
    class _NoopAuthorization:
        def verify_for_run(self, **kwargs: object) -> None:
            pass

    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=10,
            steps_per_epoch=40,
        ),
    )
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
    before = deepcopy(model.state_dict())
    with pytest.raises(
        TypeError,
        match="CoverageStateRunAuthorization",
    ):
        train_coverage_state_objective(
            model,
            optimizer,
            cache,
            schedule,
            objective=CoverageStatePairObjective.RESPONSE_JOINT,
            device="cpu",
            expected_initial_model_fingerprint=(
                coverage_state_model_fingerprint(model)
            ),
            authorization=_NoopAuthorization(),
        )
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )
    assert not optimizer.state


def test_matched_rng_contract_touches_only_selected_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    original_cpu_state = torch.get_rng_state().clone()

    class _SelectedDevice:
        def __init__(self, device: torch.device) -> None:
            self.device = device

        def __enter__(self) -> None:
            events.append(("enter", str(self.device)))

        def __exit__(self, *args: object) -> None:
            events.append(("exit", str(self.device)))

    selected = torch.device("cuda:2")
    selected_state = torch.tensor([7, 11, 13], dtype=torch.uint8)
    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda device: _SelectedDevice(torch.device(device)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: (
            events.append(("get", str(torch.device(device))))
            or selected_state
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, *, device: events.append(
            (
                "set",
                str(torch.device(device)),
                torch.equal(state, selected_state),
            )
        ),
    )
    for name in (
        "manual_seed_all",
        "get_rng_state_all",
        "set_rng_state_all",
    ):
        monkeypatch.setattr(
            torch.cuda,
            name,
            lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"{_name} must not be called")
            ),
        )
    try:
        training_module._seed_matched_rng(321, selected)
        cpu_state, cuda_state = training_module._capture_matched_rng(
            selected
        )
        assert cuda_state is not None
        training_module._restore_matched_rng(
            cpu_state,
            cuda_state,
            device=selected,
        )
    finally:
        torch.set_rng_state(original_cpu_state)
    assert events == [
        ("enter", "cuda:2"),
        ("seed", 321),
        ("exit", "cuda:2"),
        ("get", "cuda:2"),
        ("set", "cuda:2", True),
    ]


def test_epoch_callback_receives_complete_ordered_rows() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=43,
            epochs=2,
            steps_per_epoch=2,
        ),
    )
    model = _model()
    rows: list[dict[str, object]] = []
    result = train_coverage_state_objective(
        model,
        torch.optim.SGD(model.parameters(), lr=1.0e-2),
        cache,
        schedule,
        objective=CoverageStatePairObjective.SEPARABLE_ENDPOINT,
        device="cpu",
        expected_initial_model_fingerprint=(
            coverage_state_model_fingerprint(model)
        ),
        epoch_callback=rows.append,
    )
    assert [row["epoch"] for row in rows] == [0, 1]
    assert rows == list(result.epoch_logs)


def test_matched_runner_freezes_all_fairness_coordinates() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=2,
        ),
    )
    seen: list[tuple[str, int]] = []
    result = train_matched_coverage_state_objectives(
        CoverageStateLevelSetConfig(
            feature_channels=2,
            feature_stride=TOY_STRIDE,
            width=4,
        ),
        cache,
        schedule,
        config=CoverageStateMatchedTrainingConfig(seed=42),
        device="cpu",
        epoch_callback=lambda objective, row: seen.append(
            (objective, int(row["epoch"]))
        ),
    )
    assert tuple(value.objective for value in result.results) == (
        "response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    assert {
        value.initial_model_fingerprint for value in result.results
    } == {result.common_initial_model_fingerprint}
    assert {
        value.schedule_fingerprint for value in result.results
    } == {schedule.schedule_fingerprint}
    assert all(value.completed_updates == 2 for value in result.results)
    assert all(value.finite_state_audits == 3 for value in result.results)
    assert len(
        {value.device_cache_fingerprint for value in result.results}
    ) == 1
    assert seen == [
        ("response_joint", 0),
        ("identity_joint", 0),
        ("separable_endpoint", 0),
    ]
    payload = result.canonical_payload()
    assert all(payload["fairness"].values())
    assert "model_contract" not in payload
    result.models[0][1].phase_projection.bias.data.add_(1.0)
    with pytest.raises(ValueError, match="model/result binding"):
        result.verify_unchanged()


def test_completion_rooted_matched_runner_replaces_only_candidate() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=2,
        ),
    )
    result = train_matched_coverage_state_completion_rooted_objectives(
        CoverageStateLevelSetConfig(
            feature_channels=2,
            feature_stride=TOY_STRIDE,
            width=4,
        ),
        cache,
        schedule,
        config=CoverageStateMatchedTrainingConfig(seed=42),
        device="cpu",
    )
    assert tuple(value.objective for value in result.results) == (
        "completion_rooted_response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    assert result.canonical_payload()["objective_suite"] == [
        "completion_rooted_response_joint",
        "identity_joint",
        "separable_endpoint",
    ]
    assert all(result.canonical_payload()["fairness"].values())
    assert len(
        {value.initial_model_fingerprint for value in result.results}
    ) == 1
    assert len(
        {value.schedule_fingerprint for value in result.results}
    ) == 1


def test_support_oriented_matched_runner_replaces_only_candidate() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=2,
        ),
    )
    result = train_matched_coverage_state_support_oriented_objectives(
        CoverageStateLevelSetConfig(
            feature_channels=2,
            feature_stride=TOY_STRIDE,
            width=4,
        ),
        cache,
        schedule,
        config=CoverageStateMatchedTrainingConfig(seed=42),
        device="cpu",
    )
    assert tuple(value.objective for value in result.results) == (
        "support_oriented_response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    assert result.canonical_payload()["objective_suite"] == [
        "support_oriented_response_joint",
        "identity_joint",
        "separable_endpoint",
    ]
    assert all(result.canonical_payload()["fairness"].values())
    assert len(
        {value.initial_model_fingerprint for value in result.results}
    ) == 1
    assert len(
        {value.schedule_fingerprint for value in result.results}
    ) == 1


def test_phase_preserving_matched_runner_uses_one_shared_ppce_model() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=2,
        ),
    )
    model_config = CoverageStatePhasePreservingConfig(
        feature_channels=2,
        feature_stride=TOY_STRIDE,
        width=4,
    )
    result = (
        train_matched_coverage_state_phase_preserving_support_oriented_objectives(
            model_config,
            cache,
            schedule,
            config=CoverageStateMatchedTrainingConfig(seed=42),
            device="cpu",
        )
    )
    assert tuple(value.objective for value in result.results) == (
        "support_oriented_response_joint",
        "identity_joint",
        "separable_endpoint",
    )
    assert all(
        isinstance(model, CURELitePhasePreservingCoverageStateLevelSet)
        for _, model in result.models
    )
    assert all(
        model.config == model_config for _, model in result.models
    )
    assert {
        value.initial_model_fingerprint for value in result.results
    } == {result.common_initial_model_fingerprint}
    assert {
        value.schedule_fingerprint for value in result.results
    } == {schedule.schedule_fingerprint}
    assert {
        tuple(model.input_projection.weight.shape)
        for _, model in result.models
    } == {
        (
            model_config.width,
            (
                model_config.feature_channels
                + model_config.phase_occupancy_channels
            ),
            3,
            3,
        )
    }
    payload = result.canonical_payload()
    assert all(payload["fairness"].values())
    assert payload["model_contract"]["model_class"].endswith(
        ".CURELitePhasePreservingCoverageStateLevelSet"
    )
    assert payload["model_contract"]["config_class"].endswith(
        ".CoverageStatePhasePreservingConfig"
    )
    assert payload["model_contract"]["parameter_count"] == (
        model_config.expected_parameter_count
    )
    assert payload["model_contract"]["config"]["coverage_policy"] == (
        model_config.coverage_policy
    )
    assert payload["fairness"]["same_model_class"] is True
    assert payload["fairness"]["same_model_config"] is True
    assert payload["fairness"]["same_parameter_count"] is True
    assert payload["fairness"]["same_parameter_shapes"] is True


def test_matched_runner_prepares_one_shared_device_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=2,
        ),
    )
    prepare_calls = 0
    original = training_module.prepare_coverage_state_device_cache

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        training_module,
        "prepare_coverage_state_device_cache",
        counted_prepare,
    )
    result = train_matched_coverage_state_objectives(
        CoverageStateLevelSetConfig(
            feature_channels=2,
            feature_stride=TOY_STRIDE,
            width=4,
        ),
        cache,
        schedule,
        config=CoverageStateMatchedTrainingConfig(seed=42),
        device="cpu",
    )
    assert prepare_calls == 1
    assert len(
        {value.device_cache_fingerprint for value in result.results}
    ) == 1


def test_matched_runner_uses_exact_outer_and_fast_inner_cache_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=2,
        ),
    )
    calls: list[tuple[bool, bool]] = []
    original = (
        training_module.CoverageStateDeviceCache.verify_unchanged
    )

    def tracked(
        self,
        *,
        verify_content: bool = True,
        verify_source: bool = True,
    ) -> None:
        calls.append((verify_content, verify_source))
        original(
            self,
            verify_content=verify_content,
            verify_source=verify_source,
        )

    monkeypatch.setattr(
        training_module.CoverageStateDeviceCache,
        "verify_unchanged",
        tracked,
    )
    train_matched_coverage_state_objectives(
        CoverageStateLevelSetConfig(
            feature_channels=2,
            feature_stride=TOY_STRIDE,
            width=4,
        ),
        cache,
        schedule,
        config=CoverageStateMatchedTrainingConfig(seed=42),
        device="cpu",
    )
    assert calls == [
        (False, False),
        (True, False),
        (False, False),
        (False, False),
        (False, False),
        (True, False),
    ]
