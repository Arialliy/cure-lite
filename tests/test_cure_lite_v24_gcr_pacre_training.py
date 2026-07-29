from __future__ import annotations

from pathlib import Path

import pytest
import torch

import cure_lite_v24.training as training
from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    COVERAGE_STATE_FORMAL_SCOPE,
    CoverageStateRunAuthorization,
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
    coverage_state_pair_objective_policy,
)
from cure_lite_v24.factory import build_gcr_pacre_training_model
from cure_lite_v24.gcr_pacre import (
    CURELiteGatedCommonResidualPACRELevelSet,
    CoverageStateGCRPACREConfig,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


def _model_config() -> CoverageStateGCRPACREConfig:
    return CoverageStateGCRPACREConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )


def _cache_and_schedule(
    *,
    seed: int,
    epochs: int,
    steps_per_epoch: int = 40,
):
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=seed,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        ),
    )
    return cache, schedule


class _ToyAuthorization(CoverageStateRunAuthorization):
    def __init__(self, expected_scope: str) -> None:
        self.expected_scope = expected_scope
        self.calls = 0

    def verify_for_run(self, *, cache, schedule, scope) -> None:
        assert schedule.cache_fingerprint == cache.cache_fingerprint
        assert scope == self.expected_scope
        self.calls += 1


def _fake_public_trainer(call_log: list[dict[str, object]]):
    def run(
        model,
        optimizer,
        cache,
        schedule,
        *,
        objective,
        device,
        expected_initial_model_fingerprint,
        authorization,
        epoch_callback,
    ):
        assert type(model) is CURELiteGatedCommonResidualPACRELevelSet
        assert type(optimizer) is torch.optim.Adam
        assert optimizer.state == {}
        assert objective is CoverageStatePairObjective.PMOPE_JOINT
        assert coverage_state_model_fingerprint(model) == (
            expected_initial_model_fingerprint
        )
        scope = (
            COVERAGE_STATE_FORMAL_SCOPE
            if schedule.config.epochs == 800
            else COVERAGE_STATE_BOUNDED_SCOPE
        )
        authorization.verify_for_run(
            cache=cache,
            schedule=schedule,
            scope=scope,
        )
        optimizer_fingerprint = coverage_state_optimizer_config_fingerprint(
            model,
            optimizer,
        )
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters(), start=1):
                parameter.add_(index * 1.0e-4)
        updates = schedule.config.updates
        result = CoverageStateTrainingResult(
            objective=CoverageStatePairObjective.PMOPE_JOINT.value,
            objective_policy=coverage_state_pair_objective_policy(
                CoverageStatePairObjective.PMOPE_JOINT
            ),
            seed=schedule.config.seed,
            epochs=schedule.config.epochs,
            steps_per_epoch=schedule.config.steps_per_epoch,
            completed_updates=updates,
            schedule_fingerprint=schedule.schedule_fingerprint,
            cache_fingerprint=cache.cache_fingerprint,
            execution_device=str(torch.device(device)),
            device_cache_fingerprint=stable_fingerprint(
                {
                    "toy": "generated-only",
                    "seed": schedule.config.seed,
                }
            ),
            device_cache_resident_bytes=1,
            optimizer_config_fingerprint=optimizer_fingerprint,
            initial_model_fingerprint=(
                expected_initial_model_fingerprint
            ),
            final_model_fingerprint=coverage_state_model_fingerprint(
                model
            ),
            epoch_logs=tuple(
                {"epoch": epoch}
                for epoch in range(schedule.config.epochs)
            ),
            first_nonzero_gradient_update=tuple(
                (name, 0) for name, _ in model.named_parameters()
            ),
            forward_calls=updates,
            backward_calls=updates,
            optimizer_steps=updates,
            logical_state_evaluations=updates * 12,
            finite_state_audits=updates + 1,
        )
        call_log.append(
            {
                "model": model,
                "optimizer": optimizer,
                "cache": cache,
                "schedule": schedule,
                "authorization": authorization,
                "epoch_callback": epoch_callback,
            }
        )
        return result

    return run


def test_oof_core_binds_fresh_bytes_cache_schedule_and_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, schedule = _cache_and_schedule(seed=42, epochs=10)
    authorization = _ToyAuthorization(COVERAGE_STATE_BOUNDED_SCOPE)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        training,
        "train_coverage_state_objective",
        _fake_public_trainer(calls),
    )

    bundle = training.train_gcr_pacre_pmope_candidate(
        _model_config(),
        cache,
        schedule,
        role=training.GCR_PACRE_ROLE_OOF,
        seed=42,
        authorization=authorization,
    )

    assert len(calls) == 1
    assert authorization.calls == 1
    assert bundle.model is calls[0]["model"]
    assert bundle.receipt.role == "oof"
    assert bundle.receipt.completed_updates == 400
    assert bundle.receipt.cache_fingerprint == cache.cache_fingerprint
    assert bundle.receipt.schedule_fingerprint == (
        schedule.schedule_fingerprint
    )
    assert tuple(
        (row.name, row.shape, row.byte_count)
        for row in bundle.receipt.initial_parameters
    ) == (
        ("joint_state_weight", (4, 6, 5, 5), 4 * 6 * 5 * 5 * 4),
        ("joint_hidden_bias", (4,), 4 * 4),
        ("scalar_energy_weight", (4,), 4 * 4),
    )

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        initial = build_gcr_pacre_training_model(_model_config())
    assert tuple(
        row.content_fingerprint
        for row in bundle.receipt.initial_parameters
    ) == tuple(
        tensor_content_fingerprint(parameter)
        for parameter in initial.parameters()
    )
    assert bundle.receipt.D_V_execution_authorized is False
    assert bundle.receipt.D_V_payload_accessed is False
    assert (
        bundle.receipt.eligible_for_future_D_V_authorization_after_all_external_prerequisites
        is False
    )
    assert bundle.receipt.from_scratch is True
    assert bundle.receipt.resume_allowed is False
    assert bundle.receipt.automatic_retry_allowed is False
    assert bundle.receipt.checkpoint_policy == "final_only"
    bundle.verify_unchanged()

    with torch.no_grad():
        bundle.model.scalar_energy_weight.add_(1.0)
    with pytest.raises(ValueError, match="model/receipt binding changed"):
        bundle.verify_unchanged()


def test_v24_private_step_trace_preserves_r2_public_trainer_bytes() -> None:
    cache, schedule = _cache_and_schedule(seed=42, epochs=10)
    public_trainer_path = (
        Path(training.__file__).resolve().parents[1]
        / "cure_lite/experiment/coverage_state_training.py"
    )
    expected_public_sha = (
        "acbeec94db308eb3fb3dadaa741c80af5980b0b53ad001568cdeaaf18d316b6f"
    )
    rows: list[dict[str, object]] = []
    bundle = training.train_gcr_pacre_pmope_candidate(
        _model_config(),
        cache,
        schedule,
        role=training.GCR_PACRE_ROLE_BOUNDED,
        seed=42,
        authorization=_ToyAuthorization(COVERAGE_STATE_BOUNDED_SCOPE),
        update_callback=lambda row: rows.append(dict(row)),
    )

    assert file_sha256(public_trainer_path) == expected_public_sha
    assert len(rows) == 400
    assert [row["optimizer_step_counter"] for row in rows] == list(
        range(1, 401)
    )
    assert [row["selection_fingerprint"] for row in rows] == [
        selection.selection_fingerprint
        for selection in schedule.selections
    ]
    assert all(
        all(
            isinstance(row[field], str) and len(row[field]) == 64
            for field in (
                "parameter_state_digest",
                "optimizer_state_digest",
            )
        )
        for row in rows
    )
    assert bundle.training_result.completed_updates == 400


def test_formal_seed43_is_training_integrity_only_and_never_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, schedule = _cache_and_schedule(seed=43, epochs=800)
    authorization = _ToyAuthorization(COVERAGE_STATE_FORMAL_SCOPE)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        training,
        "train_coverage_state_objective",
        _fake_public_trainer(calls),
    )

    bundle = training.train_gcr_pacre_pmope_candidate(
        _model_config(),
        cache,
        schedule,
        role=training.GCR_PACRE_ROLE_TRAINING_INTEGRITY,
        seed=43,
        authorization=authorization,
    )

    receipt = bundle.receipt
    assert len(calls) == 1
    assert receipt.completed_updates == 32_000
    assert receipt.training_invocations == 1
    assert receipt.selection_effect == "none"
    assert receipt.may_replace_seed42_primary is False
    assert (
        receipt.eligible_for_future_D_V_authorization_after_all_external_prerequisites
        is False
    )
    assert (
        receipt.eligible_for_future_D_T_authorization_after_all_external_prerequisites
        is False
    )
    assert receipt.D_V_execution_authorized is False
    assert receipt.D_T_execution_authorized is False
    assert receipt.D_V_payload_accessed is False
    assert receipt.D_T_payload_accessed is False


def test_role_seed_budget_and_authorization_fail_closed_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = training.GCRPACRETrainingPolicy(
        role=training.GCR_PACRE_ROLE_PRIMARY,
        seed=42,
    )
    integrity = training.GCRPACRETrainingPolicy(
        role=training.GCR_PACRE_ROLE_TRAINING_INTEGRITY,
        seed=43,
    )
    assert (
        primary.canonical_payload()[
            "eligible_for_future_D_V_authorization_after_all_external_prerequisites"
        ]
        is True
    )
    assert (
        integrity.canonical_payload()[
            "eligible_for_future_D_V_authorization_after_all_external_prerequisites"
        ]
        is False
    )
    for seed, role in (
        (43, training.GCR_PACRE_ROLE_OOF),
        (43, training.GCR_PACRE_ROLE_PRIMARY),
        (42, training.GCR_PACRE_ROLE_TRAINING_INTEGRITY),
        (42.0, training.GCR_PACRE_ROLE_OOF),
        (True, training.GCR_PACRE_ROLE_OOF),
    ):
        with pytest.raises(ValueError, match="seed/role"):
            training.GCRPACRETrainingPolicy(  # type: ignore[arg-type]
                role=role,
                seed=seed,
            )

    cache, wrong_budget = _cache_and_schedule(
        seed=42,
        epochs=1,
        steps_per_epoch=1,
    )
    allocated = False

    def forbidden_factory(*args, **kwargs):
        del args, kwargs
        nonlocal allocated
        allocated = True
        raise AssertionError("model allocation must not be reached")

    monkeypatch.setattr(
        training,
        "build_gcr_pacre_training_model",
        forbidden_factory,
    )
    with pytest.raises(PermissionError, match="role budget"):
        training.train_gcr_pacre_pmope_candidate(
            _model_config(),
            cache,
            wrong_budget,
            role=training.GCR_PACRE_ROLE_OOF,
            seed=42,
            authorization=_ToyAuthorization(
                COVERAGE_STATE_BOUNDED_SCOPE
            ),
        )
    with pytest.raises(TypeError, match="authorization"):
        training.train_gcr_pacre_pmope_candidate(
            _model_config(),
            cache,
            wrong_budget,
            role=training.GCR_PACRE_ROLE_OOF,
            seed=42,
            authorization=object(),  # type: ignore[arg-type]
        )
    assert allocated is False


def test_receipt_binds_only_the_three_v24_training_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, schedule = _cache_and_schedule(seed=42, epochs=10)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        training,
        "train_coverage_state_objective",
        _fake_public_trainer(calls),
    )
    bundle = training.train_gcr_pacre_pmope_candidate(
        _model_config(),
        cache,
        schedule,
        role=training.GCR_PACRE_ROLE_BOUNDED,
        seed=42,
        authorization=_ToyAuthorization(COVERAGE_STATE_BOUNDED_SCOPE),
    )
    package_root = Path(training.__file__).resolve().parent
    assert dict(bundle.receipt.source_hashes) == {
        "cure_lite_v24/gcr_pacre.py": file_sha256(
            package_root / "gcr_pacre.py"
        ),
        "cure_lite_v24/factory.py": file_sha256(
            package_root / "factory.py"
        ),
        "cure_lite_v24/training.py": file_sha256(
            package_root / "training.py"
        ),
    }
    assert len(bundle.receipt.receipt_fingerprint) == 64
    assert len(bundle.bundle_fingerprint) == 64
