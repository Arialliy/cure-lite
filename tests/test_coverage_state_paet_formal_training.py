from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import cure_lite.experiment.coverage_state_paet_formal_training as formal
import cure_lite.experiment.coverage_state_paet_formal_artifacts as artifacts
from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
    coverage_state_formal_exposure_gate,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    COVERAGE_STATE_FORMAL_SCOPE,
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
    make_training_scalar_cache,
)


def test_repository_bounded_seal_binds_all_artifacts_and_source_closure():
    seal = (
        formal
        .load_repository_coverage_state_paet_bounded_artifact_seal()
    )
    seal.verify_unchanged()
    payload = seal.payload

    assert payload["complete_fingerprint"] == (
        formal.COVERAGE_STATE_PAET_BOUNDED_COMPLETE_FINGERPRINT
    )
    assert len(payload["artifact_binding"]) == 17
    assert payload["structural_advancement_passed"] is True
    assert payload["generic_population_gate_passed"] is False
    assert payload["bounded_evidence_is_performance"] is False
    assert payload["dataset_free_gate_passed"] is True
    assert payload["D_R_gate_passed"] is True
    assert payload["full_D_R_scalar_cache_fingerprint"] == (
        formal.COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
    )
    assert payload["source_closure"]["source_file_count"] == 44
    assert payload["source_closure"][
        "current_inherited_implementation_matches_closure"
    ] is True
    assert payload["formal_frozen_coordinates"] == {
        "seed": 42,
        "epochs": 800,
        "steps_per_epoch": 40,
        "updates": 32_000,
        "full_D_R_scalar_cache_fingerprint": (
            formal.COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        ),
        "schedule_fingerprint": (
            formal.COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
        ),
        "exposure_gate_fingerprint": (
            formal.COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT
        ),
        "initial_model_fingerprint": (
            formal.COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT
        ),
        "coordinates_require_runtime_reconstruction": True,
    }
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    payload["structural_advancement_passed"] = False
    assert seal.structural_advancement_passed is True


class _FakeRealInputs:
    def __init__(self, cache, *, build_fingerprint, source_fingerprint):
        self.scalar_cache = cache
        self.build_fingerprint = build_fingerprint
        self.source_binding = SimpleNamespace(
            dataset="IRSTD-1K",
            split="D_R",
            binding_fingerprint=source_fingerprint,
        )
        self.verify_calls = 0

    def verify_unchanged(self) -> None:
        self.verify_calls += 1
        self.scalar_cache.verify_unchanged()


def _fake_bounded_audit(real_inputs) -> dict[str, object]:
    return {
        "schema_version": formal.COVERAGE_STATE_PAET_BOUNDED_SEAL_SCHEMA,
        "run_id": formal.COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
        "run_repo_path": (
            formal.COVERAGE_STATE_PAET_BOUNDED_RUN_REPO_PATH
        ),
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
        "D_R_gate_evidence_fingerprint": "f" * 64,
        "real_inputs_build_fingerprint": (
            real_inputs.build_fingerprint
        ),
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
            "feature_channels": 2,
            "feature_stride": 2,
            "width": 4,
            "parameter_count": 608,
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


@pytest.fixture(scope="module")
def toy_environment():
    patch = pytest.MonkeyPatch()
    cache = make_bounded_training_scalar_cache()
    real_inputs = _FakeRealInputs(
        cache,
        build_fingerprint="1" * 64,
        source_fingerprint="2" * 64,
    )
    config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig.formal(seed=42),
    )
    exposure = coverage_state_formal_exposure_gate(cache, schedule)
    assert exposure["all_pass"] is True
    audit = _fake_bounded_audit(real_inputs)
    audit_calls = {"count": 0}

    def audit_once():
        audit_calls["count"] += 1
        return audit

    patch.setattr(formal, "CoverageStateRealDRInputs", _FakeRealInputs)
    patch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS",
        2,
    )
    patch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE",
        2,
    )
    patch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_WIDTH",
        4,
    )
    patch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT",
        config.expected_parameter_count,
    )
    patch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT",
        cache.cache_fingerprint,
    )
    patch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT",
        schedule.schedule_fingerprint,
    )
    patch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT",
        exposure["gate_fingerprint"],
    )
    patch.setattr(
        formal,
        "COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT",
        formal._formal_initial_model_fingerprint(config),
    )
    patch.setattr(
        formal,
        "_audit_repository_bounded_evidence",
        audit_once,
    )
    seal = (
        formal
        .load_repository_coverage_state_paet_bounded_artifact_seal()
    )
    try:
        yield SimpleNamespace(
            cache=cache,
            real_inputs=real_inputs,
            config=config,
            schedule=schedule,
            exposure=exposure,
            audit=audit,
            audit_calls=audit_calls,
            seal=seal,
        )
    finally:
        patch.undo()


def _make_authorization(env):
    implementation = formal._current_formal_implementation_binding()
    exposure_checks = tuple(
        sorted(env.exposure["checks"].items())
    )
    model_fingerprint = stable_fingerprint(
        formal._formal_model_config_payload(env.config)
    )
    initial_fingerprint = formal._formal_initial_model_fingerprint(
        env.config
    )
    implementation_fingerprint = stable_fingerprint(
        dict(implementation)
    )
    static = formal._formal_authorization_static_binding_payload(
        run_id=formal.COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        real_inputs=env.real_inputs,
        bounded_artifact_seal=env.seal,
        schedule=env.schedule,
        exposure_gate_fingerprint=env.exposure["gate_fingerprint"],
        exposure_gate_checks=exposure_checks,
        model_config_fingerprint=model_fingerprint,
        expected_parameter_count=env.config.expected_parameter_count,
        expected_initial_model_fingerprint=initial_fingerprint,
        formal_implementation_binding=implementation,
        formal_implementation_fingerprint=(
            implementation_fingerprint
        ),
    )
    preparation_seal = formal._FormalPreparationSeal(
        real_inputs=env.real_inputs,
        scalar_cache=env.cache,
        bounded_artifact_seal=env.seal,
        schedule=env.schedule,
        static_binding_fingerprint=stable_fingerprint(static),
    )
    return formal.CoverageStatePAETFormal800Authorization(
        run_id=formal.COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        real_inputs=env.real_inputs,
        bounded_artifact_seal=env.seal,
        schedule=env.schedule,
        exposure_gate_fingerprint=env.exposure["gate_fingerprint"],
        exposure_gate_checks=exposure_checks,
        model_config_fingerprint=model_fingerprint,
        expected_parameter_count=env.config.expected_parameter_count,
        expected_initial_model_fingerprint=initial_fingerprint,
        formal_implementation_binding=implementation,
        formal_implementation_fingerprint=implementation_fingerprint,
        _preparation_seal=preparation_seal,
        _run_once_seal=formal._FormalRunOnceSeal(),
    )


def _count_source_cache_verifications(
    monkeypatch: pytest.MonkeyPatch,
    source: CoverageStateScalarCache,
) -> dict[str, int]:
    calls = {"count": 0}
    original = CoverageStateScalarCache.verify_unchanged

    def counted(self):
        if self is source:
            calls["count"] += 1
        return original(self)

    monkeypatch.setattr(
        CoverageStateScalarCache,
        "verify_unchanged",
        counted,
    )
    return calls


def test_prepare_uses_complete_cache_and_exact_formal_budget(
    toy_environment,
    monkeypatch: pytest.MonkeyPatch,
):
    env = toy_environment
    env.real_inputs.verify_calls = 0
    env.audit_calls["count"] = 0
    cache_verify_calls = _count_source_cache_verifications(
        monkeypatch,
        env.cache,
    )
    exposure_calls = 0
    original_exposure = formal.coverage_state_formal_exposure_gate

    def exposure_once(cache, schedule):
        nonlocal exposure_calls
        exposure_calls += 1
        return original_exposure(cache, schedule)

    monkeypatch.setattr(
        formal,
        "coverage_state_formal_exposure_gate",
        exposure_once,
    )
    authorization = (
        formal.prepare_coverage_state_paet_formal_800_authorization(
            env.real_inputs,
            env.config,
            bounded_artifact_seal=env.seal,
        )
    )
    payload = authorization.canonical_payload()

    assert authorization.schedule.cache_fingerprint == (
        env.cache.cache_fingerprint
    )
    assert authorization.schedule is not env.schedule
    assert authorization.schedule.config == (
        CoverageStateScheduleConfig.formal(seed=42)
    )
    assert len(authorization.schedule.selections) == 32_000
    assert payload["budget"] == {
        "seed": 42,
        "epochs": 800,
        "steps_per_epoch": 40,
        "updates": 32_000,
        "objectives": 1,
    }
    assert payload["structural_advancement_passed"] is True
    assert payload["generic_population_gate_passed"] is False
    assert "bounded_gate_passed" not in payload
    assert payload["bounded_evidence_interpretation"] == (
        "structural_advancement_only_not_performance"
    )
    assert payload["formal_D_R_training_authorized"] is True
    assert payload["training_contract"]["from_scratch"] is True
    assert payload["training_contract"]["checkpoint_policy"] == (
        "final_model_only"
    )
    assert payload["field_threshold_hex"] == 0.0.hex()
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["performance_claim_supported"] is False
    assert env.real_inputs.verify_calls == 1
    assert env.audit_calls["count"] == 1
    assert exposure_calls == 1
    assert cache_verify_calls["count"] == 1


def test_authorization_rejects_wrong_scope_cache_schedule_and_model(
    toy_environment,
):
    env = toy_environment
    authorization = _make_authorization(env)
    authorization.verify_for_run(
        cache=env.cache,
        schedule=env.schedule,
        scope=COVERAGE_STATE_FORMAL_SCOPE,
    )
    assert authorization._run_once_seal.claimed is True
    with pytest.raises(PermissionError, match="already consumed"):
        authorization.verify_for_run(
            cache=env.cache,
            schedule=env.schedule,
            scope=COVERAGE_STATE_FORMAL_SCOPE,
        )
    wrong_scope_authorization = _make_authorization(env)
    with pytest.raises(PermissionError, match="scope"):
        wrong_scope_authorization.verify_for_run(
            cache=env.cache,
            schedule=env.schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
        )
    other_cache = make_training_scalar_cache()
    wrong_cache_authorization = _make_authorization(env)
    with pytest.raises(PermissionError, match="scope"):
        wrong_cache_authorization.verify_for_run(
            cache=other_cache,
            schedule=env.schedule,
            scope=COVERAGE_STATE_FORMAL_SCOPE,
        )
    equivalent_schedule = build_coverage_state_training_schedule(
        env.cache,
        CoverageStateScheduleConfig.formal(seed=42),
    )
    assert (
        equivalent_schedule.schedule_fingerprint
        == env.schedule.schedule_fingerprint
    )
    wrong_schedule_authorization = _make_authorization(env)
    with pytest.raises(PermissionError, match="scope"):
        wrong_schedule_authorization.verify_for_run(
            cache=env.cache,
            schedule=equivalent_schedule,
            scope=COVERAGE_STATE_FORMAL_SCOPE,
        )
    with pytest.raises(PermissionError, match="model config"):
        authorization.verify_model_config(
            CoverageStatePhaseAlignedEvidenceTransportConfig(
                feature_channels=2,
                feature_stride=2,
                width=5,
            )
        )


def test_forged_seal_or_nonfull_real_inputs_are_rejected(
    toy_environment,
    monkeypatch: pytest.MonkeyPatch,
):
    env = toy_environment
    with pytest.raises(ValueError, match="fingerprint"):
        replace(env.seal, audit_fingerprint="0" * 64)

    changed = dict(env.audit)
    changed["structural_advancement_passed"] = False
    monkeypatch.setattr(
        formal,
        "_audit_repository_bounded_evidence",
        lambda: changed,
    )
    with pytest.raises(RuntimeError, match="evidence changed"):
        env.seal.verify_unchanged()

    monkeypatch.setattr(
        formal,
        "_audit_repository_bounded_evidence",
        lambda: env.audit,
    )
    wrong_inputs = _FakeRealInputs(
        make_training_scalar_cache(),
        build_fingerprint=env.real_inputs.build_fingerprint,
        source_fingerprint=(
            env.real_inputs.source_binding.binding_fingerprint
        ),
    )
    with pytest.raises(
        (PermissionError, ValueError),
        match="formal PAET|authorization binding",
    ):
        formal.prepare_coverage_state_paet_formal_800_authorization(
            wrong_inputs,
            env.config,
            bounded_artifact_seal=env.seal,
        )


def _fake_formal_training(authorization, config):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(config)
    initial = coverage_state_model_fingerprint(model)
    assert initial == authorization.expected_initial_model_fingerprint
    with torch.no_grad():
        model.scalar_energy_weight.fill_(0.125)
    final = coverage_state_model_fingerprint(model)
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
        final_model_fingerprint=final,
        epoch_logs=tuple(
            {
                "epoch": value,
                "completed_updates": (value + 1) * 40,
                "objective": objective,
                "selection_sequence_fingerprint": stable_fingerprint(
                    {"toy_formal_epoch": value}
                ),
                "mean_factual_miss/loss": 0.1,
                "mean_factual_no_miss/loss": 0.2,
                "mean_pair/loss": 0.3,
                "mean_total": 0.6,
                "mean_gradient_l2_norm": 0.4,
            }
            for value in range(800)
        ),
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
        cache_fingerprint=(
            authorization.real_inputs.scalar_cache.cache_fingerprint
        ),
        results=(row,),
        models=((objective, model),),
    )


def test_runner_is_single_from_scratch_call_and_D_R_only(
    toy_environment,
    monkeypatch: pytest.MonkeyPatch,
):
    env = toy_environment
    authorization = _make_authorization(env)
    calls: list[dict[str, object]] = []
    training = _fake_formal_training(authorization, env.config)
    env.real_inputs.verify_calls = 0
    env.audit_calls["count"] = 0
    cache_verify_calls = _count_source_cache_verifications(
        monkeypatch,
        env.cache,
    )
    exposure_calls = 0
    original_exposure = formal.coverage_state_formal_exposure_gate

    def train(model_config, cache, schedule, **kwargs):
        kwargs["authorization"].verify_for_run(
            cache=cache,
            schedule=schedule,
            scope=COVERAGE_STATE_FORMAL_SCOPE,
        )
        calls.append(
            {
                "model_config": model_config,
                "cache": cache,
                "schedule": schedule,
                **kwargs,
            }
        )
        return training

    def exposure_once(cache, schedule):
        nonlocal exposure_calls
        exposure_calls += 1
        return original_exposure(cache, schedule)

    monkeypatch.setattr(
        formal,
        "train_matched_coverage_state_paet_bfa_pmope_objectives",
        train,
    )
    monkeypatch.setattr(
        formal,
        "_deterministic_execution",
        lambda device: nullcontext(),
    )
    monkeypatch.setattr(
        formal,
        "coverage_state_formal_exposure_gate",
        exposure_once,
    )
    callback = lambda objective, row: None
    result = formal.run_coverage_state_paet_bfa_pmope_formal_800(
        authorization,
        env.config,
        device="cuda:0",
        epoch_callback=callback,
    )

    assert len(calls) == 1
    assert calls[0]["cache"] is env.cache
    assert calls[0]["schedule"] is env.schedule
    assert calls[0]["authorization"] is authorization
    assert calls[0]["epoch_callback"] is callback
    assert calls[0]["config"] == CoverageStateMatchedTrainingConfig(
        seed=42
    )
    assert result.training_complete
    assert result.final_model is training.models[0][1]
    payload = result.canonical_payload()
    assert payload["structural_advancement_passed"] is True
    assert payload["generic_population_gate_passed"] is False
    assert payload["training_contract"]["from_scratch"] is True
    assert payload["training_contract"]["resume_allowed"] is False
    assert payload["training_contract"]["automatic_retry_allowed"] is False
    assert payload["training_contract"]["checkpoint_policy"] == (
        "final_model_only"
    )
    assert payload["field_threshold_hex"] == 0.0.hex()
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["performance_evaluation_performed"] is False
    assert env.real_inputs.verify_calls == 1
    assert env.audit_calls["count"] == 1
    assert exposure_calls == 1
    assert cache_verify_calls["count"] == 1

    with pytest.raises(PermissionError, match="already consumed"):
        formal.run_coverage_state_paet_bfa_pmope_formal_800(
            authorization,
            env.config,
            device="cuda:0",
        )
    assert len(calls) == 1
    assert env.real_inputs.verify_calls == 1
    assert env.audit_calls["count"] == 1
    assert exposure_calls == 1
    assert cache_verify_calls["count"] == 1


def test_actual_formal_result_round_trips_final_artifact_contract(
    toy_environment,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """Exercise the real result payload instead of a hand-written surrogate."""

    env = toy_environment
    authorization = _make_authorization(env)
    authorization.claim_once()
    training = _fake_formal_training(authorization, env.config)
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
    for name, value in (
        ("COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS", 2),
        ("COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE", 2),
        ("COVERAGE_STATE_PAET_FORMAL_WIDTH", 4),
        (
            "COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT",
            env.config.expected_parameter_count,
        ),
        (
            "COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT",
            env.cache.cache_fingerprint,
        ),
        (
            "COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT",
            authorization.schedule.schedule_fingerprint,
        ),
    ):
        monkeypatch.setattr(artifacts, name, value)

    target = tmp_path / "actual-formal-result"
    saved_fingerprint = artifacts.save_coverage_state_paet_formal_artifact(
        target,
        result,
    )
    loaded = artifacts.load_coverage_state_paet_formal_artifact(
        target,
        expected_authorization_fingerprint=(
            authorization.authorization_fingerprint
        ),
        expected_result_fingerprint=result.result_fingerprint,
    )

    assert loaded.artifact_fingerprint == saved_fingerprint
    assert loaded.formal_result_payload == result.canonical_payload()
    assert loaded.formal_result_payload["training_contract"] == {
        "from_scratch": True,
        "process_local_single_attempt_claim": True,
        "cross_process_output_claim_required": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "checkpoint_policy": "final_model_only",
        "intermediate_checkpoint_saved": False,
        "optimizer_state_saved": False,
    }
    loaded.verify_unchanged()


def test_failed_attempt_cannot_be_retried(
    toy_environment,
    monkeypatch: pytest.MonkeyPatch,
):
    env = toy_environment
    authorization = _make_authorization(env)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        formal,
        "train_matched_coverage_state_paet_bfa_pmope_objectives",
        fail,
    )
    monkeypatch.setattr(
        formal,
        "_deterministic_execution",
        lambda device: nullcontext(),
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        formal.run_coverage_state_paet_bfa_pmope_formal_800(
            authorization,
            env.config,
            device="cuda:0",
        )
    with pytest.raises(PermissionError, match="already consumed"):
        formal.run_coverage_state_paet_bfa_pmope_formal_800(
            authorization,
            env.config,
            device="cuda:0",
        )
    assert calls == 1


def test_cpu_is_rejected_before_consuming_authorization(
    toy_environment,
):
    env = toy_environment
    authorization = _make_authorization(env)
    with pytest.raises(PermissionError, match="cuda:0"):
        formal.run_coverage_state_paet_bfa_pmope_formal_800(
            authorization,
            env.config,
            device="cpu",
        )
    assert authorization._run_once_seal.claimed is False
