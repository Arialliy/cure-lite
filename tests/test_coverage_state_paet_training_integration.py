from __future__ import annotations

import torch

import cure_lite.experiment.coverage_state_training as training_module
from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.coverage_state_phase_preserving import (
    build_coverage_state_level_set,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.experiment.coverage_state_training import (
    CoverageStateMatchedTrainingConfig,
    CoverageStateMatchedTrainingResult,
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
    train_matched_coverage_state_paet_bfa_pmope_objectives,
)
from cure_lite.train.coverage_state_fused_step import (
    COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES,
    coverage_state_pair_objective_policy,
)
from tests_v15.coverage_state_test_helpers import TOY_STRIDE
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


def _toy_config() -> CoverageStatePhaseAlignedEvidenceTransportConfig:
    return CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=2,
        feature_stride=TOY_STRIDE,
        width=4,
    )


def test_factory_dispatches_paet_before_its_bfa_parent_family() -> None:
    config = _toy_config()
    model = build_coverage_state_level_set(config)

    assert type(model) is CURELitePhaseAlignedEvidenceTransportLevelSet
    assert model.config is config
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        config.expected_parameter_count
    )


def test_paet_wrapper_routes_only_frozen_pmope(
    monkeypatch,
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
    model_config = _toy_config()
    matched_config = CoverageStateMatchedTrainingConfig(seed=42)
    sentinel = object()
    captured: dict[str, object] = {}

    def train_suite(
        actual_model_config,
        actual_cache,
        actual_schedule,
        **kwargs,
    ):
        captured.update(
            {
                "model_config": actual_model_config,
                "cache": actual_cache,
                "schedule": actual_schedule,
                **kwargs,
            }
        )
        return sentinel

    monkeypatch.setattr(
        training_module,
        "_train_matched_coverage_state_objective_suite",
        train_suite,
    )
    result = train_matched_coverage_state_paet_bfa_pmope_objectives(
        model_config,
        cache,
        schedule,
        config=matched_config,
        device="cpu",
    )

    assert result is sentinel
    assert captured == {
        "model_config": model_config,
        "cache": cache,
        "schedule": schedule,
        "config": matched_config,
        "device": "cpu",
        "objectives": COVERAGE_STATE_PMOPE_MATCHED_OBJECTIVES,
        "authorization": None,
        "epoch_callback": None,
    }


def test_paet_uses_the_existing_three_parameter_gradient_contract() -> None:
    model = build_coverage_state_level_set(_toy_config())

    training_module._validate_coverage_state_gradient_latency(
        model,
        {
            "scalar_energy_weight": 0,
            "joint_state_weight": 2,
            "joint_hidden_bias": 2,
        },
    )


def test_paet_wrapper_rejects_bfa_config() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=1,
            steps_per_epoch=1,
        ),
    )
    bfa = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=2,
        feature_stride=TOY_STRIDE,
        width=4,
    )

    try:
        train_matched_coverage_state_paet_bfa_pmope_objectives(
            bfa,  # type: ignore[arg-type]
            cache,
            schedule,
            config=CoverageStateMatchedTrainingConfig(seed=42),
            device=torch.device("cpu"),
        )
    except TypeError as error:
        assert "CoverageStatePhaseAlignedEvidenceTransportConfig" in str(
            error
        )
    else:
        raise AssertionError("the PAET wrapper accepted a BFA config")


def test_paet_result_names_candidate_without_changing_pmope() -> None:
    model = build_coverage_state_level_set(_toy_config())
    fingerprint = coverage_state_model_fingerprint(model)
    result = CoverageStateTrainingResult(
        objective="pmope_joint",
        objective_policy=coverage_state_pair_objective_policy(
            "pmope_joint"
        ),
        seed=42,
        epochs=1,
        steps_per_epoch=1,
        completed_updates=1,
        schedule_fingerprint="schedule",
        cache_fingerprint="cache",
        execution_device="cpu",
        device_cache_fingerprint="d" * 64,
        device_cache_resident_bytes=1,
        optimizer_config_fingerprint="optimizer",
        initial_model_fingerprint=fingerprint,
        final_model_fingerprint=fingerprint,
        epoch_logs=({},),
        first_nonzero_gradient_update=(),
        forward_calls=1,
        backward_calls=1,
        optimizer_steps=1,
        logical_state_evaluations=12,
        finite_state_audits=2,
    )
    matched = CoverageStateMatchedTrainingResult(
        config=CoverageStateMatchedTrainingConfig(seed=42),
        common_initial_model_fingerprint=fingerprint,
        schedule_fingerprint="schedule",
        cache_fingerprint="cache",
        results=(result,),
        models=(("pmope_joint", model),),
    )

    payload = matched.canonical_payload()
    assert payload["objective_suite"] == ["pmope_joint"]
    assert payload["fairness"] == {
        "candidate_model": "PAET-BFA",
        "single_candidate_only": True,
        "same_initial_state": True,
        "same_schedule": True,
        "same_endpoints": True,
        "same_model": True,
        "same_optimizer": True,
        "same_device_cache": True,
        "same_compute_budget": True,
        "same_natural_branches": True,
        "historical_controls_retrained": False,
        "historical_v20_objective_reused": True,
        "allowed_difference_from_sealed_v20": (
            "predeclared_field_equation_only"
        ),
        "same_model_class": True,
        "same_model_config": True,
        "same_parameter_count": True,
        "same_parameter_shapes": True,
    }
