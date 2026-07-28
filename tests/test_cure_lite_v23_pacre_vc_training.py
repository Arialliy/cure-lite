from __future__ import annotations

from pathlib import Path

import pytest
import torch

import cure_lite_v22.factory as v22_factory
import cure_lite_v23.training as training
from cure_lite.cache.schema import file_sha256
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.experiment.coverage_state_training import (
    CoverageStateRunAuthorization,
)
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
)
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


def _config() -> CoverageStatePACREVerifierCorrectedConfig:
    return CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )


def _cache_and_schedule(*, epochs: int = 1, steps: int = 3):
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=epochs,
            steps_per_epoch=steps,
        ),
    )
    return cache, schedule


def test_training_constructs_only_exact_v23_and_fresh_adam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, schedule = _cache_and_schedule()
    calls: list[tuple[object, object]] = []
    real_train = training.train_coverage_state_objective

    monkeypatch.setattr(
        v22_factory,
        "build_pacre_training_model",
        lambda *args, **kwargs: pytest.fail(
            "the v22 exact-type factory must not construct v23"
        ),
    )

    def train_spy(model, optimizer, actual_cache, actual_schedule, **kwargs):
        assert type(model) is CURELitePACREVerifierCorrectedLevelSet
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.state == {}
        assert actual_cache is cache
        assert actual_schedule is schedule
        assert kwargs["objective"] is CoverageStatePairObjective.PMOPE_JOINT
        calls.append((model, optimizer))
        return real_train(
            model,
            optimizer,
            actual_cache,
            actual_schedule,
            **kwargs,
        )

    monkeypatch.setattr(
        training,
        "train_coverage_state_objective",
        train_spy,
    )
    bundle = training.train_pacre_vc_pmope_candidate(
        _config(),
        cache,
        schedule,
        device="cpu",
    )

    assert len(calls) == 1
    assert bundle.model is calls[0][0]
    assert bundle.training_result.completed_updates == 3
    assert bundle.training_result.forward_calls == 3
    assert bundle.receipt.model_fqcn == training.PACRE_VC_MODEL_FQCN
    assert bundle.receipt.config_fqcn == training.PACRE_VC_CONFIG_FQCN
    assert bundle.receipt.parameter_count == 608
    assert bundle.receipt.initial_model_fingerprint != (
        bundle.receipt.final_model_fingerprint
    )
    bundle.verify_unchanged()


def test_training_receipt_binds_v23_sources_and_fixed_seed42_policy() -> None:
    cache, schedule = _cache_and_schedule(steps=3)
    bundle = training.train_pacre_vc_pmope_candidate(
        _config(),
        cache,
        schedule,
    )
    package_root = Path(training.__file__).resolve().parent

    assert dict(bundle.receipt.source_hashes) == {
        "cure_lite_v23/pacre_vc.py": file_sha256(
            package_root / "pacre_vc.py"
        ),
        "cure_lite_v23/factory.py": file_sha256(
            package_root / "factory.py"
        ),
        "cure_lite_v23/training.py": file_sha256(
            package_root / "training.py"
        ),
    }
    assert bundle.receipt.seed == 42
    assert bundle.receipt.objective == "pmope_joint"
    assert training.PACRE_VC_PMOPE_TRAINING_CONFIG.canonical_payload()[
        "optimizer_fqcn"
    ] == "torch.optim.adam.Adam"
    with pytest.raises(ValueError, match="fixes seed"):
        training.PACREVCPMOPETrainingConfig(seed=43)


def test_protected_budgets_fail_closed_before_model_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, bounded = _cache_and_schedule(epochs=10, steps=40)
    _, formal = _cache_and_schedule(epochs=800, steps=40)

    class _WrongAuthorization(CoverageStateRunAuthorization):
        def verify_for_run(self, *, cache, schedule, scope) -> None:
            del cache, schedule, scope

    allocated = False

    def forbidden_resolve(device):
        del device
        nonlocal allocated
        allocated = True
        raise AssertionError("allocation must not be reached")

    monkeypatch.setattr(training, "_resolve_device", forbidden_resolve)
    with pytest.raises(TypeError, match="exact authorization"):
        training.train_pacre_vc_pmope_candidate(
            _config(),
            cache,
            bounded,
            authorization=_WrongAuthorization(),
            device="cuda:0",
        )
    with pytest.raises(PermissionError, match="Formal800"):
        training.train_pacre_vc_pmope_candidate(
            _config(),
            cache,
            formal,
            device="cuda:0",
        )
    assert allocated is False
