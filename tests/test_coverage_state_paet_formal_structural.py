from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import cure_lite.experiment.coverage_state_paet_formal_structural as structural
import cure_lite.experiment.coverage_state_paet_formal_training as formal
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
    coverage_state_formal_exposure_gate,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
)
from cure_lite.experiment.coverage_state_training import (
    CoverageStateMatchedTrainingConfig,
    CoverageStateMatchedTrainingResult,
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
)
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
    coverage_state_pair_objective_policy,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)


class _SyntheticRealInputs:
    def __init__(self, cache, *, build_fingerprint, source_fingerprint):
        self.scalar_cache = cache
        self.build_fingerprint = build_fingerprint
        self.source_binding = SimpleNamespace(
            dataset="IRSTD-1K",
            split="D_R",
            binding_fingerprint=source_fingerprint,
        )

    def verify_unchanged(self) -> None:
        self.scalar_cache.verify_unchanged()


def _synthetic_bounded_audit(
    real_inputs: _SyntheticRealInputs,
    *,
    config: CoverageStatePhaseAlignedEvidenceTransportConfig,
    evidence_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": formal.COVERAGE_STATE_PAET_BOUNDED_SEAL_SCHEMA,
        "run_id": formal.COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
        "run_repo_path": formal.COVERAGE_STATE_PAET_BOUNDED_RUN_REPO_PATH,
        "complete_fingerprint": "a" * 64,
        "complete_file_sha256": "b" * 64,
        "artifact_binding": {"COMPLETE.json": "b" * 64},
        "artifact_binding_fingerprint": "c" * 64,
        "bounded_result_fingerprint": "d" * 64,
        "structural_advancement_passed": True,
        "generic_population_gate_passed": False,
        "bounded_evidence_is_performance": False,
        "dataset_free_gate_passed": True,
        "dataset_free_receipt_fingerprint": "e" * 64,
        "D_R_gate_passed": True,
        "D_R_gate_evidence_fingerprint": evidence_fingerprint,
        "real_inputs_build_fingerprint": real_inputs.build_fingerprint,
        "source_binding_fingerprint": (
            real_inputs.source_binding.binding_fingerprint
        ),
        "full_D_R_scalar_cache_fingerprint": (
            real_inputs.scalar_cache.cache_fingerprint
        ),
        "full_D_R_scalar_cache_counts": (
            real_inputs.scalar_cache.canonical_payload()["counts"]
        ),
        "model": {
            "feature_channels": config.feature_channels,
            "feature_stride": config.feature_stride,
            "width": config.width,
            "parameter_count": config.expected_parameter_count,
            "candidate_objective": "pmope_joint",
            "candidate_objective_policy": formal.CSLF_PMOPE_POLICY,
            "field_threshold_hex": 0.0.hex(),
        },
        "source_closure": {
            "source_file_count": 1,
            "current_inherited_implementation_matches_closure": True,
        },
        "D_V_accessed": False,
        "D_T_accessed": False,
        "performance_claim_supported": False,
    }


def _synthetic_training(authorization, config):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(config)
    initial = coverage_state_model_fingerprint(model)
    assert initial == authorization.expected_initial_model_fingerprint
    with torch.no_grad():
        model.scalar_energy_weight.fill_(0.125)
    objective = CoverageStatePairObjective.PMOPE_JOINT.value
    row = CoverageStateTrainingResult(
        objective=objective,
        objective_policy=coverage_state_pair_objective_policy(objective),
        seed=42,
        epochs=800,
        steps_per_epoch=40,
        completed_updates=32_000,
        schedule_fingerprint=authorization.schedule.schedule_fingerprint,
        cache_fingerprint=(
            authorization.real_inputs.scalar_cache.cache_fingerprint
        ),
        execution_device="cuda:0",
        device_cache_fingerprint="3" * 64,
        device_cache_resident_bytes=1,
        optimizer_config_fingerprint="4" * 64,
        initial_model_fingerprint=initial,
        final_model_fingerprint=coverage_state_model_fingerprint(model),
        epoch_logs=tuple({"epoch": value} for value in range(800)),
        first_nonzero_gradient_update=(
            ("joint_hidden_bias", 1),
            ("joint_state_weight", 1),
            ("scalar_energy_weight", 0),
        ),
        forward_calls=32_000,
        backward_calls=32_000,
        optimizer_steps=32_000,
        logical_state_evaluations=384_000,
        finite_state_audits=32_001,
    )
    return CoverageStateMatchedTrainingResult(
        config=CoverageStateMatchedTrainingConfig(seed=42),
        common_initial_model_fingerprint=initial,
        schedule_fingerprint=authorization.schedule.schedule_fingerprint,
        cache_fingerprint=authorization.real_inputs.scalar_cache.cache_fingerprint,
        results=(row,),
        models=((objective, model),),
    )


def _synthetic_formal_result_and_population(monkeypatch: pytest.MonkeyPatch):
    """Build an authenticated synthetic result without invoking training."""

    cache = make_bounded_training_scalar_cache()
    config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    real_inputs = _SyntheticRealInputs(
        cache,
        build_fingerprint="1" * 64,
        source_fingerprint="2" * 64,
    )
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig.formal(seed=42),
    )
    exposure = coverage_state_formal_exposure_gate(cache, schedule)
    evidence_fingerprint = "5" * 64
    audit = _synthetic_bounded_audit(
        real_inputs,
        config=config,
        evidence_fingerprint=evidence_fingerprint,
    )
    monkeypatch.setattr(formal, "CoverageStateRealDRInputs", _SyntheticRealInputs)
    monkeypatch.setattr(formal, "COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS", 2)
    monkeypatch.setattr(formal, "COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE", 2)
    monkeypatch.setattr(formal, "COVERAGE_STATE_PAET_FORMAL_WIDTH", 4)
    monkeypatch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT",
        config.expected_parameter_count,
    )
    monkeypatch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT",
        formal._formal_initial_model_fingerprint(config),
    )
    monkeypatch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT",
        cache.cache_fingerprint,
    )
    monkeypatch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT",
        schedule.schedule_fingerprint,
    )
    monkeypatch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT",
        exposure["gate_fingerprint"],
    )
    monkeypatch.setattr(formal, "_audit_repository_bounded_evidence", lambda: audit)

    seal = formal.load_repository_coverage_state_paet_bounded_artifact_seal()
    authorization = formal.prepare_coverage_state_paet_formal_800_authorization(
        real_inputs,
        config,
        bounded_artifact_seal=seal,
    )
    authorization.claim_once()
    training = _synthetic_training(authorization, config)
    result = formal.CoverageStatePAETFormal800RunResult(
        authorization=authorization,
        training=training,
        training_invocations=1,
        checks=formal._formal_result_checks(
            authorization,
            training,
            training_invocations=1,
        ),
    )
    population = build_coverage_state_bounded_population(cache, seed=42)
    monkeypatch.setattr(
        structural,
        "COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT",
        cache.cache_fingerprint,
    )
    monkeypatch.setattr(
        structural,
        "COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT",
        population.cache.cache_fingerprint,
    )
    monkeypatch.setattr(
        structural,
        "COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT",
        population.population_fingerprint,
    )
    monkeypatch.setattr(
        structural,
        "COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT",
        real_inputs.source_binding.binding_fingerprint,
    )
    monkeypatch.setattr(
        structural,
        "COVERAGE_STATE_PAET_V21_DR_GATE_EVIDENCE_FINGERPRINT",
        evidence_fingerprint,
    )
    return result, population


def test_authenticated_synthetic_path_preserves_all_gate_semantics(
    monkeypatch: pytest.MonkeyPatch,
):
    formal_result, population = _synthetic_formal_result_and_population(monkeypatch)
    result = structural.evaluate_coverage_state_paet_formal_structural_retention(
        formal_result,
        population,
        device="cpu",
    )
    payload = result.canonical_payload()

    assert payload["runtime_splits"] == ["D_R"]
    assert payload["bounded400_structural_advancement_passed"] is True
    assert (
        payload["post_formal_structural_retention_passed"]
        is result.post_formal_structural_retention_passed
    )
    assert (
        payload["generic_population_gate_passed"]
        is result.generic_population_gate_passed
    )
    assert payload["performance_status"] == "NOT_EVALUATED"
    assert payload["performance_gate_passed"] is None
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False


def test_public_evaluator_rejects_forged_result_and_wrong_population_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    formal_result, population = _synthetic_formal_result_and_population(monkeypatch)
    with pytest.raises(TypeError, match="exact CoverageStatePAETFormal800RunResult"):
        structural.evaluate_coverage_state_paet_formal_structural_retention(
            SimpleNamespace(), population, device="cpu"
        )

    substituted = replace(
        population,
        cache=population.source_cache,
        bounded_cache_fingerprint=population.source_cache.cache_fingerprint,
    )
    with pytest.raises(PermissionError, match="substituted coordinates"):
        structural.evaluate_coverage_state_paet_formal_structural_retention(
            formal_result, substituted, device="cpu"
        )


def test_public_evaluator_rejects_changed_formal_model(
    monkeypatch: pytest.MonkeyPatch,
):
    formal_result, population = _synthetic_formal_result_and_population(monkeypatch)
    with torch.no_grad():
        formal_result.final_model.scalar_energy_weight.add_(1.0)
    with pytest.raises(ValueError, match="trained model/result binding"):
        structural.evaluate_coverage_state_paet_formal_structural_retention(
            formal_result, population, device="cpu"
        )


def test_structural_receipt_constructor_cannot_bypass_private_factory(
    monkeypatch: pytest.MonkeyPatch,
):
    formal_result, population = _synthetic_formal_result_and_population(monkeypatch)
    result = structural.evaluate_coverage_state_paet_formal_structural_retention(
        formal_result,
        population,
        device="cpu",
    )
    with pytest.raises(PermissionError, match="not sealed"):
        replace(result, _seal=object())


def test_issued_structural_receipt_rechecks_later_model_mutation(
    monkeypatch: pytest.MonkeyPatch,
):
    formal_result, population = _synthetic_formal_result_and_population(
        monkeypatch
    )
    result = structural.evaluate_coverage_state_paet_formal_structural_retention(
        formal_result,
        population,
        device="cpu",
    )
    with torch.no_grad():
        formal_result.final_model.scalar_energy_weight.add_(1.0)
    with pytest.raises(
        (RuntimeError, ValueError),
        match="trained model/result binding|structural receipt changed",
    ):
        result.canonical_payload()
