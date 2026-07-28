from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest
import torch

import cure_lite.experiment.coverage_state_training as legacy_training
import cure_lite.coverage_state_phase_preserving as legacy_registry
import cure_lite_v22.training as pacre_training
from cure_lite.cache.schema import file_sha256
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.experiment.coverage_state_training import (
    CoverageStateMatchedTrainingResult,
    CoverageStateRunAuthorization,
    coverage_state_model_fingerprint,
)
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
)
from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
)
from cure_lite_v22.training import (
    PACRE_CONFIG_FQCN,
    PACRE_MODEL_FQCN,
    PACRE_PARAMETER_NAMES,
    PACRE_PMOPE_OBJECTIVE,
    PACRE_PMOPE_TRAINING_CONFIG,
    PACRE_SOURCE_PATHS,
    PACREPMOPETrainingBundle,
    PACREPMOPETrainingConfig,
    train_pacre_pmope_candidate,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


def _model_config() -> CoverageStatePACREConfig:
    return CoverageStatePACREConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )


def _cache_and_schedule(*, seed: int = 42):
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=seed,
            epochs=1,
            steps_per_epoch=3,
        ),
    )
    return cache, schedule


def _unexpected_legacy_call(*args, **kwargs):
    del args, kwargs
    raise AssertionError("legacy builder/trainer must not be called")


def test_single_candidate_training_bypasses_legacy_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, schedule = _cache_and_schedule()
    monkeypatch.setattr(
        legacy_registry,
        "build_coverage_state_level_set",
        _unexpected_legacy_call,
    )
    monkeypatch.setattr(
        legacy_training,
        "build_coverage_state_level_set",
        _unexpected_legacy_call,
    )
    monkeypatch.setattr(
        legacy_training,
        "train_matched_coverage_state_paet_bfa_pmope_objectives",
        _unexpected_legacy_call,
    )

    real_factory = pacre_training.build_pacre_training_model
    real_public_train = pacre_training.train_coverage_state_objective
    constructed = []
    public_calls: list[dict[str, object]] = []

    def factory_spy(config):
        model = real_factory(config)
        constructed.append(model)
        return model

    def public_train_spy(
        model,
        optimizer,
        actual_cache,
        actual_schedule,
        **kw,
    ):
        public_calls.append(
            {
                "model": model,
                "optimizer": optimizer,
                "cache": actual_cache,
                "schedule": actual_schedule,
                **kw,
            }
        )
        return real_public_train(
            model,
            optimizer,
            actual_cache,
            actual_schedule,
            **kw,
        )

    monkeypatch.setattr(
        pacre_training,
        "build_pacre_training_model",
        factory_spy,
    )
    monkeypatch.setattr(
        pacre_training,
        "train_coverage_state_objective",
        public_train_spy,
    )
    bundle = train_pacre_pmope_candidate(
        _model_config(),
        cache,
        schedule,
        device="cpu",
    )

    assert type(bundle) is PACREPMOPETrainingBundle
    assert not isinstance(bundle, CoverageStateMatchedTrainingResult)
    assert len(constructed) == 1
    assert bundle.model is constructed[0]
    assert (
        type(bundle.model)
        is
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
    )
    assert len(public_calls) == 1
    call = public_calls[0]
    assert call["model"] is bundle.model
    assert isinstance(call["optimizer"], torch.optim.Adam)
    assert call["cache"] is cache
    assert call["schedule"] is schedule
    assert call["objective"] is CoverageStatePairObjective.PMOPE_JOINT
    assert call["device"] == torch.device("cpu")

    result = bundle.training_result
    assert result.objective == PACRE_PMOPE_OBJECTIVE
    assert result.seed == 42
    assert result.completed_updates == 3
    assert result.forward_calls == 3
    assert result.backward_calls == 3
    assert result.optimizer_steps == 3
    assert result.initial_model_fingerprint != (
        result.final_model_fingerprint
    )
    assert coverage_state_model_fingerprint(bundle.model) == (
        result.final_model_fingerprint
    )
    assert dict(result.first_nonzero_gradient_update) == {
        "joint_hidden_bias": 1,
        "joint_state_weight": 1,
        "scalar_energy_weight": 0,
    }


def test_receipt_binds_exact_structure_sources_and_compute() -> None:
    cache, schedule = _cache_and_schedule()
    bundle = train_pacre_pmope_candidate(
        _model_config(),
        cache,
        schedule,
    )
    receipt = bundle.receipt

    assert receipt.model_fqcn == PACRE_MODEL_FQCN
    assert receipt.config_fqcn == PACRE_CONFIG_FQCN
    assert receipt.seed == 42
    assert receipt.objective == PACRE_PMOPE_OBJECTIVE
    assert receipt.training_config_fingerprint == (
        PACRE_PMOPE_TRAINING_CONFIG.config_fingerprint
    )
    assert json.loads(receipt.training_config_json) == (
        PACRE_PMOPE_TRAINING_CONFIG.canonical_payload()
    )
    contract = json.loads(receipt.model_contract_json)
    assert contract["model_class"] == PACRE_MODEL_FQCN
    assert contract["config_class"] == PACRE_CONFIG_FQCN
    assert contract["parameter_count"] == 608
    assert tuple(row.name for row in receipt.parameter_topology) == (
        PACRE_PARAMETER_NAMES
    )
    assert tuple(row.shape for row in receipt.parameter_topology) == (
        (4, 6, 5, 5),
        (4,),
        (4,),
    )
    assert receipt.parameter_count == 608
    assert receipt.initial_model_fingerprint == (
        bundle.training_result.initial_model_fingerprint
    )
    assert receipt.final_model_fingerprint == (
        bundle.training_result.final_model_fingerprint
    )
    assert receipt.forward_calls == 3
    assert receipt.completed_updates == 3

    package_root = Path(pacre_training.__file__).resolve().parent
    expected_sources = {
        "cure_lite_v22/pacre.py": package_root / "pacre.py",
        "cure_lite_v22/factory.py": package_root / "factory.py",
        "cure_lite_v22/training.py": package_root / "training.py",
    }
    assert tuple(expected_sources) == PACRE_SOURCE_PATHS
    assert dict(receipt.source_hashes) == {
        name: file_sha256(path)
        for name, path in expected_sources.items()
    }
    assert len(receipt.receipt_fingerprint) == 64
    assert len(bundle.bundle_fingerprint) == 64
    bundle.verify_unchanged()


def test_bundle_and_receipt_are_frozen_and_detect_model_mutation() -> None:
    cache, schedule = _cache_and_schedule()
    bundle = train_pacre_pmope_candidate(
        _model_config(),
        cache,
        schedule,
    )

    with pytest.raises(FrozenInstanceError):
        bundle.receipt.seed = 43  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        bundle.model = object()  # type: ignore[misc]

    with torch.no_grad():
        bundle.model.scalar_energy_weight.add_(1.0)
    with pytest.raises(
        ValueError,
        match="trained model/receipt binding changed",
    ):
        bundle.verify_unchanged()


def test_training_rejects_wrong_model_or_training_configuration() -> None:
    cache, schedule = _cache_and_schedule()
    v21 = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    with pytest.raises(
        TypeError,
        match="exact type CoverageStatePACREConfig",
    ):
        train_pacre_pmope_candidate(v21, cache, schedule)

    with pytest.raises(ValueError, match="fixes seed"):
        PACREPMOPETrainingConfig(seed=43)
    with pytest.raises(ValueError, match="fixes learning_rate"):
        PACREPMOPETrainingConfig(learning_rate=0.002)

    other_cache, other_schedule = _cache_and_schedule(seed=43)
    del other_cache
    with pytest.raises(ValueError, match="schedule seed differ"):
        train_pacre_pmope_candidate(
            _model_config(),
            cache,
            other_schedule,
        )


def test_protected_training_rejects_non_pacre_authorization_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=10,
            steps_per_epoch=40,
        ),
    )

    class _NominalAuthorization(CoverageStateRunAuthorization):
        def verify_for_run(self, *, cache, schedule, scope) -> None:
            del cache, schedule, scope

    reached_allocation = False

    def forbidden_device(value):
        del value
        nonlocal reached_allocation
        reached_allocation = True
        raise AssertionError("device allocation path was reached")

    monkeypatch.setattr(
        pacre_training,
        "_resolve_device",
        forbidden_device,
    )
    with pytest.raises(
        TypeError,
        match="exact authorization",
    ):
        train_pacre_pmope_candidate(
            _model_config(),
            cache,
            schedule,
            authorization=_NominalAuthorization(),
            device="cuda:0",
        )
    assert reached_allocation is False
