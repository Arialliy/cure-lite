from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import json
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch

import cure_lite_v22.bounded_runner as bounded_runner
import cure_lite_v22.dr_gate as dr_gate
import cure_lite_v22.training as pacre_training
from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRInputs,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    CoverageStateTrainingResult,
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
)
from cure_lite.experiment.coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
)
from cure_lite.frozen_base import module_state_fingerprint
from cure_lite_v22.bounded_runner import (
    CoverageStatePACREBoundedRunAuthorization,
    CoverageStatePACREBoundedRunResult,
    prepare_pacre_bounded_run_authorization,
    run_pacre_pmope_bounded_400,
)
from cure_lite_v22.dataset_free import (
    run_pacre_dataset_free_gate,
)
from cure_lite_v22.decision import (
    PACRE_BOUNDED_RUN_ID,
    CoverageStatePACREBoundedDecision,
)
from cure_lite_v22.dr_gate import (
    CoverageStatePACREDRGateReceipt,
    recompute_pacre_dr_checks,
)
from cure_lite_v22.factory import build_pacre_training_model
from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
)
from cure_lite_v22.training import (
    PACRE_CONFIG_FQCN,
    PACRE_MODEL_FQCN,
    PACRE_OPTIMIZER_FQCN,
    PACRE_PMOPE_OBJECTIVE,
    PACRE_PMOPE_TRAINING_CONFIG,
    PACRE_PMOPE_TRAINING_SCHEMA,
    PACREPMOPETrainingBundle,
    PACREPMOPETrainingReceipt,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)
from tools import run_cure_lite_v22_pacre_bounded_400 as pacre_cli


class _GeneratedRealInputs(CoverageStateRealDRInputs):
    """Minimal generated-only graph accepted by the live receipt verifier."""

    def __init__(self, scalar_cache) -> None:
        object.__setattr__(
            self,
            "source_binding",
            SimpleNamespace(split="D_R", dataset="generated"),
        )
        object.__setattr__(self, "scalar_cache", scalar_cache)
        object.__setattr__(
            self,
            "build_fingerprint",
            stable_fingerprint(
                {
                    "generated_real_inputs": (
                        scalar_cache.cache_fingerprint
                    )
                }
            ),
        )

    def verify_unchanged(self) -> None:
        self.scalar_cache.verify_unchanged()
        if self.scalar_cache.raw_catalog.split != "D_R":
            raise RuntimeError("generated fixture left D_R")


@pytest.fixture(scope="module")
def generated_graph():
    source_cache = make_bounded_training_scalar_cache()
    population = build_coverage_state_bounded_population(source_cache)
    preflight = prepare_coverage_state_bounded_preflight(population)
    real_inputs = _GeneratedRealInputs(source_cache)
    model_config = CoverageStatePACREConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    return preflight, real_inputs, model_config


@pytest.fixture(autouse=True)
def isolated_attempt_registry():
    """Each test represents a fresh process-level official run context."""

    with bounded_runner._ATTEMPT_REGISTRY_LOCK:
        bounded_runner._ATTEMPT_REGISTRY.clear()
    yield
    with bounded_runner._ATTEMPT_REGISTRY_LOCK:
        bounded_runner._ATTEMPT_REGISTRY.clear()


def _exact_output_attempt_receipt(
    *,
    dataset_free_receipt_fingerprint: str = "b" * 64,
) -> dict[str, object]:
    runtime = {
        "device": pacre_cli.FROZEN_DEVICE,
        "CUDA_VISIBLE_DEVICES": pacre_cli.FROZEN_VISIBLE_GPU,
        "CUBLAS_WORKSPACE_CONFIG": (
            pacre_cli.FROZEN_CUBLAS_WORKSPACE_CONFIG
        ),
        "temperature_wrapper_repo_path": (
            pacre_cli.TEMPERATURE_WRAPPER_REPO_PATH
        ),
        "temperature_wrapper_file_sha256": (
            pacre_cli.TEMPERATURE_WRAPPER_FILE_SHA256
        ),
        "pause_temperature_c": (
            pacre_cli.FROZEN_PAUSE_TEMPERATURE_C
        ),
        "resume_temperature_c": (
            pacre_cli.FROZEN_RESUME_TEMPERATURE_C
        ),
    }
    body: dict[str, object] = {
        "schema_version": (
            bounded_runner.PACRE_BOUNDED_ATTEMPT_RECEIPT_SCHEMA
        ),
        "run_id": PACRE_BOUNDED_RUN_ID,
        "output_repo_path": (
            bounded_runner.PACRE_BOUNDED_OUTPUT_REPO_PATH
        ),
        "config_fingerprint": "a" * 64,
        "runtime": runtime,
        "candidate": "PACRE-v22",
        "objective": PACRE_PMOPE_OBJECTIVE,
        "budget": {
            "seed": 42,
            "epochs": 10,
            "steps_per_epoch": 40,
            "updates": 400,
        },
        "process_identity": (
            bounded_runner.pacre_bounded_process_identity()
        ),
        "dataset_free_receipt_fingerprint": (
            dataset_free_receipt_fingerprint
        ),
        "dataset_free_invocations_before_claim": 1,
        "single_attempt": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "formal_800_authorized": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    receipt = {
        **body,
        "receipt_fingerprint": stable_fingerprint(body),
    }
    assert frozenset(receipt) == pacre_cli._ATTEMPT_FIELDS
    assert frozenset(runtime) == pacre_cli._ATTEMPT_RUNTIME_FIELDS
    return receipt


def _write_active_output_claim(
    output,
    receipt: dict[str, object],
) -> None:
    output.mkdir()
    (output / "attempt.json").write_bytes(
        (canonical_json(receipt) + "\n").encode("utf-8")
    )
    (output / ".incomplete").write_bytes(b"")


@pytest.fixture
def pacre_output_claim(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    generated_graph,
):
    """Give each test an isolated exact fixed-output attempt claim."""

    output = tmp_path / "fixed-pacre-bounded-output"
    monkeypatch.setattr(
        bounded_runner,
        "PACRE_BOUNDED_OUTPUT_PATH",
        output,
    )
    _, _, model_config = generated_graph
    dataset_free = _toy_dataset_free_receipt(model_config)
    _write_active_output_claim(
        output,
        _exact_output_attempt_receipt(
            dataset_free_receipt_fingerprint=(
                dataset_free["receipt_fingerprint"]
            ),
        ),
    )
    claim = bounded_runner.load_pacre_bounded_output_claim()
    claim.verify_unchanged()
    return claim


def _toy_dataset_free_receipt(
    model_config: CoverageStatePACREConfig,
) -> dict[str, object]:
    body = dict(run_pacre_dataset_free_gate())
    body.pop("receipt_fingerprint")
    body["parameter_count"] = model_config.expected_parameter_count
    return {**body, "receipt_fingerprint": stable_fingerprint(body)}


def _passing_probe(
    preflight,
    model_config: CoverageStatePACREConfig,
) -> dict[str, object]:
    (
        contract_json,
        contract_fingerprint,
        initial_fingerprint,
        model_fqcn,
        _,
    ) = bounded_runner._model_binding(model_config)
    population = preflight.population
    return {
        "device": "cpu",
        "execution_seed": 42,
        "model_fqcn": model_fqcn,
        "config_fqcn": PACRE_CONFIG_FQCN,
        "model_contract": json.loads(contract_json),
        "model_contract_fingerprint": contract_fingerprint,
        "initial_model_fingerprint": initial_fingerprint,
        "final_model_fingerprint": initial_fingerprint,
        "parameter_ids_preserved": True,
        "representation": {
            "target_group_count": 32,
            "target_forward_calls": 32,
            "context_state_count": 96,
            "context_forward_calls": 96,
            "all_fields_exact_pacre": True,
            "all_algebra_checks_passed": True,
            "all_target_groups_have_joint_witness": True,
            "exact_latent_collision_count": 0,
            "zero_readout_anchor_all_target_states": True,
            "fixed_readout_interaction_nonzero": True,
        },
        "gradient_path": {
            "initial_gradient_finite": {
                "joint_hidden_bias": True,
                "joint_state_weight": True,
                "scalar_energy_weight": True,
            },
            "initial_gradient_nonzero": {
                "joint_hidden_bias": False,
                "joint_state_weight": False,
                "scalar_energy_weight": True,
            },
            "readout_visible_upstream_dormant": True,
            "readout_to_upstream_cross_gradient_finite_nonzero": [
                True,
                True,
            ],
            "parameter_grad_buffers_unretained": True,
        },
        "field_direction": {
            "aggregate_descent_direction_correct": True,
            "all_roles_finite_nonzero_correct": True,
        },
        "population_fingerprint_before": (
            population.population_fingerprint
        ),
        "population_fingerprint_after": (
            population.population_fingerprint
        ),
        "cache_fingerprint_before": (
            population.cache.cache_fingerprint
        ),
        "cache_fingerprint_after": (
            population.cache.cache_fingerprint
        ),
        "global_cpu_rng_preserved": True,
        "selected_device_rng_preserved": True,
        "parameter_grad_buffers_unretained": True,
        "deterministic_execution": {
            "restored_exactly": True,
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "D_R_accessed": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def _dr_receipt(
    dataset_free_receipt,
    preflight,
    real_inputs,
    model_config,
    *,
    passed: bool = True,
) -> CoverageStatePACREDRGateReceipt:
    probe = _passing_probe(preflight, model_config)
    if not passed:
        probe["D_V_accessed"] = True
    checks = recompute_pacre_dr_checks(
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt["receipt_fingerprint"]
        ),
        real_inputs=real_inputs,
        bounded_population=preflight.population,
        probe=probe,
    )
    return CoverageStatePACREDRGateReceipt(
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt["receipt_fingerprint"]
        ),
        real_inputs_fingerprint=real_inputs.build_fingerprint,
        population_fingerprint=(
            preflight.population.population_fingerprint
        ),
        cache_fingerprint=(
            preflight.population.bounded_cache_fingerprint
        ),
        implementation_binding=dr_gate._implementation_binding(),
        probe_json=canonical_json(probe),
        checks=checks,
    )


def _authorization(
    monkeypatch: pytest.MonkeyPatch,
    generated_graph,
    pacre_output_claim,
):
    preflight, real_inputs, model_config = generated_graph
    monkeypatch.setattr(
        dr_gate,
        "PACRE_FORMAL_PARAMETER_COUNT",
        model_config.expected_parameter_count,
    )
    monkeypatch.setattr(
        dr_gate,
        "PACRE_FORMAL_FEATURE_CHANNELS",
        model_config.feature_channels,
    )
    monkeypatch.setattr(
        dr_gate,
        "PACRE_FORMAL_FEATURE_STRIDE",
        model_config.feature_stride,
    )
    monkeypatch.setattr(
        dr_gate,
        "PACRE_FORMAL_WIDTH",
        model_config.width,
    )
    dataset_free = _toy_dataset_free_receipt(model_config)
    receipt = _dr_receipt(
        dataset_free,
        preflight,
        real_inputs,
        model_config,
    )
    authorization = prepare_pacre_bounded_run_authorization(
        preflight,
        dataset_free,
        receipt,
        real_inputs,
        model_config,
        output_claim=pacre_output_claim,
        run_id=PACRE_BOUNDED_RUN_ID,
    )
    return authorization, dataset_free, receipt


def _training_bundle(
    authorization: CoverageStatePACREBoundedRunAuthorization,
) -> PACREPMOPETrainingBundle:
    model_config = authorization.model_config
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = build_pacre_training_model(model_config)
    initial = coverage_state_model_fingerprint(model)
    with torch.no_grad():
        model.scalar_energy_weight.fill_(0.125)
    final = coverage_state_model_fingerprint(model)
    result = CoverageStateTrainingResult(
        objective=PACRE_PMOPE_OBJECTIVE,
        objective_policy=bounded_runner.CSLF_PMOPE_POLICY,
        seed=42,
        epochs=10,
        steps_per_epoch=40,
        completed_updates=400,
        schedule_fingerprint=(
            authorization.preflight.schedule.schedule_fingerprint
        ),
        cache_fingerprint=(
            authorization.preflight.population.bounded_cache_fingerprint
        ),
        execution_device="cuda:0",
        device_cache_fingerprint="d" * 64,
        device_cache_resident_bytes=1,
        optimizer_config_fingerprint="e" * 64,
        initial_model_fingerprint=initial,
        final_model_fingerprint=final,
        epoch_logs=tuple({} for _ in range(10)),
        first_nonzero_gradient_update=(
            ("joint_hidden_bias", 1),
            ("joint_state_weight", 1),
            ("scalar_energy_weight", 0),
        ),
        forward_calls=400,
        backward_calls=400,
        optimizer_steps=400,
        logical_state_evaluations=4800,
        finite_state_audits=401,
    )
    contract = coverage_state_model_contract_payload(model)
    receipt = PACREPMOPETrainingReceipt(
        schema_version=PACRE_PMOPE_TRAINING_SCHEMA,
        seed=42,
        objective=PACRE_PMOPE_OBJECTIVE,
        optimizer_fqcn=PACRE_OPTIMIZER_FQCN,
        training_config_json=canonical_json(
            PACRE_PMOPE_TRAINING_CONFIG.canonical_payload()
        ),
        training_config_fingerprint=(
            PACRE_PMOPE_TRAINING_CONFIG.config_fingerprint
        ),
        model_fqcn=PACRE_MODEL_FQCN,
        config_fqcn=PACRE_CONFIG_FQCN,
        model_contract_json=canonical_json(contract),
        model_contract_fingerprint=stable_fingerprint(contract),
        source_hashes=pacre_training._source_hashes(),
        parameter_topology=pacre_training._parameter_topology(model),
        parameter_count=model_config.expected_parameter_count,
        cache_fingerprint=result.cache_fingerprint,
        schedule_fingerprint=result.schedule_fingerprint,
        optimizer_config_fingerprint=(
            result.optimizer_config_fingerprint
        ),
        initial_model_fingerprint=initial,
        final_model_fingerprint=final,
        training_result_fingerprint=result.result_fingerprint,
        completed_updates=400,
        forward_calls=400,
    )
    return PACREPMOPETrainingBundle(
        model_config=model_config,
        training_config=PACRE_PMOPE_TRAINING_CONFIG,
        model=model,
        training_result=result,
        receipt=receipt,
    )


def _diagnostic(
    authorization: CoverageStatePACREBoundedRunAuthorization,
    model,
) -> CoverageStateZeroLevelEvaluationResult:
    return CoverageStateZeroLevelEvaluationResult(
        config=CoverageStateZeroLevelEvaluationConfig(
            input_representation=(
                COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
            ),
        ),
        dataset="generated",
        split="D_R",
        cache_fingerprint=(
            authorization.preflight.population.bounded_cache_fingerprint
        ),
        checkpoint_fingerprint=module_state_fingerprint(model),
        state_ledger=(),
        natural_diagnostics=(),
        pair_diagnostics=(),
        diagnostic_state_references=0,
        unique_actual_input_states=0,
        model_forward_invocations=0,
        exact_replay_forward_invocations=0,
        reused_state_references=0,
        backward_calls=0,
        optimizer_steps=0,
        factual_miss_gate_passed=True,
        factual_no_miss_gate_passed=True,
        clean_defined_metrics_passed=True,
        clean_compact_support_gate_passed=True,
        component_null_gate_passed=True,
        identity_null_gate_passed=True,
        scalar_hidden_diagnostic_gate_passed=True,
        bounded_gate_passed=True,
        fail_closed_reasons=(),
    )


def _decision(
    diagnostic: CoverageStateZeroLevelEvaluationResult,
) -> CoverageStatePACREBoundedDecision:
    return CoverageStatePACREBoundedDecision(
        run_id=PACRE_BOUNDED_RUN_ID,
        diagnostic=diagnostic,
        reference_decision_fingerprint="f" * 64,
        checks=(("generated_gate", True),),
        population=(),
        observed=(),
        response_sign_diagnostic=(),
    )


def test_authorization_revalidates_complete_dr_graph_and_budget(
    monkeypatch: pytest.MonkeyPatch,
    generated_graph,
    pacre_output_claim,
) -> None:
    calls: list[dict[str, object]] = []
    original = CoverageStatePACREDRGateReceipt.verify_unchanged

    def verify_spy(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(
        CoverageStatePACREDRGateReceipt,
        "verify_unchanged",
        verify_spy,
    )
    authorization, dataset_free, _ = _authorization(
        monkeypatch,
        generated_graph,
        pacre_output_claim,
    )
    authorization.verify_unchanged()
    rng_before = torch.random.get_rng_state().clone()
    monkeypatch.setattr(
        torch,
        "manual_seed",
        lambda *args, **kwargs: pytest.fail(
            "model binding must not reseed every CUDA generator"
        ),
    )
    authorization.verify_model_config(authorization.model_config)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    payload = authorization.canonical_payload()

    assert calls
    assert all(
        call["dataset_free_receipt"] == dataset_free
        and call["real_inputs"] is authorization.real_inputs
        and call["bounded_population"]
        is authorization.preflight.population
        for call in calls
    )
    assert authorization.available
    assert not authorization.consumed
    assert payload["budget"] == {
        "seed": 42,
        "epochs": 10,
        "steps_per_epoch": 40,
        "updates": 400,
        "objectives": 1,
    }
    assert payload["runtime_splits"] == ["D_R"]
    assert payload["single_use"] is True
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["model"]["model_fqcn"] == PACRE_MODEL_FQCN
    assert payload["output_claim"] == (
        pacre_output_claim.canonical_payload()
    )
    assert payload["output_claim_fingerprint"] == (
        pacre_output_claim.claim_fingerprint
    )
    assert len(authorization.authorization_fingerprint) == 64


def test_implementation_binding_covers_complete_local_source_population() -> None:
    root = bounded_runner.Path(bounded_runner.__file__).resolve().parents[1]
    expected = tuple(
        sorted(
            str(path.relative_to(root))
            for package in ("cure_lite", "cure_lite_v22")
            for path in (root / package).rglob("*.py")
            if "build" not in path.relative_to(root / package).parts
        )
    )
    assert bounded_runner.PACRE_BOUNDED_IMPLEMENTATION_PATHS == expected
    paths = set(expected)
    assert {
        "cure_lite/__init__.py",
        "cure_lite/data.py",
        "cure_lite/sampling.py",
        "cure_lite/splits.py",
        "cure_lite/experiment/cache_pipeline.py",
        "cure_lite/experiment/coverage_state_bfa_dataset_free.py",
        "cure_lite/experiment/coverage_state_paet_dataset_free.py",
        "cure_lite/experiment/coverage_state_real_dr_inputs.py",
        "cure_lite/experiment/geometry_safe_catalog.py",
        "cure_lite/experiment/training_pipeline.py",
        "cure_lite_v22/bounded_runner.py",
        "cure_lite_v22/dr_gate.py",
    } <= paths
    assert all("/build/" not in path for path in paths)
    assert tuple(name for name, _ in bounded_runner._implementation_binding()) == (
        expected
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_top_level_field",
        "extra_top_level_field",
        "wrong_top_level_field",
        "missing_runtime_field",
        "extra_runtime_field",
        "wrong_runtime_field",
    ),
)
def test_output_claim_rejects_nonexact_attempt_receipt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    receipt = _exact_output_attempt_receipt()
    body = dict(receipt)
    body.pop("receipt_fingerprint")
    runtime = dict(body["runtime"])
    body["runtime"] = runtime

    if mutation == "missing_top_level_field":
        body.pop("candidate")
    elif mutation == "extra_top_level_field":
        body["unexpected"] = True
    elif mutation == "wrong_top_level_field":
        body["candidate"] = "not-PACRE-v22"
    elif mutation == "missing_runtime_field":
        runtime.pop("device")
    elif mutation == "extra_runtime_field":
        runtime["unexpected"] = True
    elif mutation == "wrong_runtime_field":
        runtime["device"] = "cpu"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    malformed = {
        **body,
        "receipt_fingerprint": stable_fingerprint(body),
    }
    output = tmp_path / mutation
    monkeypatch.setattr(
        bounded_runner,
        "PACRE_BOUNDED_OUTPUT_PATH",
        output,
    )
    _write_active_output_claim(output, malformed)

    with pytest.raises(ValueError, match="attempt|fields|invalid"):
        bounded_runner.load_pacre_bounded_output_claim()


def test_authorization_rejects_failed_dr_gate_and_wrong_run_id(
    monkeypatch: pytest.MonkeyPatch,
    generated_graph,
    pacre_output_claim,
) -> None:
    preflight, real_inputs, model_config = generated_graph
    monkeypatch.setattr(
        dr_gate,
        "PACRE_FORMAL_PARAMETER_COUNT",
        model_config.expected_parameter_count,
    )
    monkeypatch.setattr(
        dr_gate,
        "PACRE_FORMAL_FEATURE_CHANNELS",
        model_config.feature_channels,
    )
    monkeypatch.setattr(
        dr_gate,
        "PACRE_FORMAL_FEATURE_STRIDE",
        model_config.feature_stride,
    )
    monkeypatch.setattr(
        dr_gate,
        "PACRE_FORMAL_WIDTH",
        model_config.width,
    )
    dataset_free = _toy_dataset_free_receipt(model_config)
    failed = _dr_receipt(
        dataset_free,
        preflight,
        real_inputs,
        model_config,
        passed=False,
    )
    with pytest.raises(PermissionError, match="prerequisite"):
        prepare_pacre_bounded_run_authorization(
            preflight,
            dataset_free,
            failed,
            real_inputs,
            model_config,
            output_claim=pacre_output_claim,
            run_id=PACRE_BOUNDED_RUN_ID,
        )
    with pytest.raises(PermissionError, match="run_id"):
        prepare_pacre_bounded_run_authorization(
            preflight,
            dataset_free,
            _dr_receipt(
                dataset_free,
                preflight,
                real_inputs,
                model_config,
            ),
            real_inputs,
            model_config,
            output_claim=pacre_output_claim,
            run_id="wrong",
        )


def test_runner_calls_exact_pacre_stages_once_and_consumes_token(
    monkeypatch: pytest.MonkeyPatch,
    generated_graph,
    pacre_output_claim,
) -> None:
    authorization, dataset_free, dr_receipt = _authorization(
        monkeypatch,
        generated_graph,
        pacre_output_claim,
    )
    duplicate_prepare = prepare_pacre_bounded_run_authorization(
        authorization.preflight,
        dataset_free,
        dr_receipt,
        authorization.real_inputs,
        authorization.model_config,
        output_claim=pacre_output_claim,
        run_id=PACRE_BOUNDED_RUN_ID,
    )
    assert duplicate_prepare.attempt_fingerprint == (
        authorization.attempt_fingerprint
    )
    assert duplicate_prepare._attempt_token is authorization._attempt_token
    shallow_copy = copy.copy(authorization)
    replaced_copy = replace(authorization)
    training = _training_bundle(authorization)
    diagnostic = _diagnostic(authorization, training.model)
    decision = _decision(diagnostic)
    order: list[str] = []

    def train(
        model_config,
        cache,
        schedule,
        *,
        config,
        device,
        authorization: CoverageStatePACREBoundedRunAuthorization,
    ):
        order.append("training")
        assert type(model_config) is CoverageStatePACREConfig
        assert cache is authorization.preflight.population.cache
        assert schedule is authorization.preflight.schedule
        assert config is PACRE_PMOPE_TRAINING_CONFIG
        assert device == torch.device("cuda:0")
        authorization.verify_reserved_for_training(
            model_config=model_config,
            cache=cache,
            schedule=schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
            device="cuda:0",
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            initial_model = build_pacre_training_model(model_config)
        initial_fingerprint = coverage_state_model_fingerprint(
            initial_model
        )
        authorization.consume_for_training(
            model=initial_model,
            model_config=model_config,
            cache=cache,
            schedule=schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
            device="cuda:0",
            objective=PACRE_PMOPE_OBJECTIVE,
            initial_model_fingerprint=initial_fingerprint,
        )
        authorization.verify_for_run(
            cache=cache,
            schedule=schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
        )
        return training

    def evaluate(model, cache, *, device, config):
        order.append("zero_level")
        assert (
            type(model)
            is
            CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
        )
        assert cache is authorization.preflight.population.cache
        assert device == torch.device("cuda:0")
        assert config.input_representation == (
            COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        )
        return diagnostic

    def decide(actual, *, run_id):
        if "decision" not in order:
            order.append("decision")
        assert actual is diagnostic
        assert run_id == PACRE_BOUNDED_RUN_ID
        return decision

    monkeypatch.setattr(
        bounded_runner,
        "_deterministic_execution",
        lambda device: nullcontext(),
    )
    monkeypatch.setattr(
        bounded_runner,
        "train_pacre_pmope_candidate",
        train,
    )
    monkeypatch.setattr(
        bounded_runner,
        "evaluate_coverage_state_zero_level_checkpoint",
        evaluate,
    )
    monkeypatch.setattr(
        bounded_runner,
        "decide_coverage_state_pacre_bounded",
        decide,
    )
    monkeypatch.setattr(
        bounded_runner,
        "_model_parameter_devices",
        lambda model: ("cuda:0",),
    )
    result = run_pacre_pmope_bounded_400(
        authorization,
        authorization.model_config,
        run_id=PACRE_BOUNDED_RUN_ID,
        device="cuda:0",
    )

    assert type(result) is CoverageStatePACREBoundedRunResult
    assert order == ["training", "zero_level", "decision"]
    assert authorization.consumed
    assert shallow_copy.consumed
    assert replaced_copy.consumed
    assert duplicate_prepare.consumed
    assert not authorization.available
    assert result.bounded_gate_passed
    assert result.formal800_eligible
    payload = result.canonical_payload()
    assert payload["runtime_splits"] == ["D_R"]
    assert payload["execution_invocations"] == {
        "training": 1,
        "zero_level_evaluation": 1,
        "decision": 1,
    }
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["formal_800_authorized"] is False
    assert len(result.result_fingerprint) == 64

    with pytest.raises(PermissionError, match="no longer available"):
        run_pacre_pmope_bounded_400(
            authorization,
            authorization.model_config,
            run_id=PACRE_BOUNDED_RUN_ID,
            device="cuda:0",
        )
    assert order == ["training", "zero_level", "decision"]
    for duplicate in (
        shallow_copy,
        replaced_copy,
        duplicate_prepare,
    ):
        with pytest.raises(PermissionError, match="no longer available"):
            duplicate.claim_for_training(
                model_config=duplicate.model_config,
                cache=duplicate.preflight.population.cache,
                schedule=duplicate.preflight.schedule,
                scope=COVERAGE_STATE_BOUNDED_SCOPE,
                device="cuda:0",
            )


def test_runner_rejects_wrong_config_or_device_before_training(
    monkeypatch: pytest.MonkeyPatch,
    generated_graph,
    pacre_output_claim,
) -> None:
    authorization, _, _ = _authorization(
        monkeypatch,
        generated_graph,
        pacre_output_claim,
    )
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("training must not be reached")

    monkeypatch.setattr(
        bounded_runner,
        "train_pacre_pmope_candidate",
        forbidden,
    )
    with pytest.raises(PermissionError, match="cuda:0"):
        run_pacre_pmope_bounded_400(
            authorization,
            authorization.model_config,
            run_id=PACRE_BOUNDED_RUN_ID,
            device="cpu",
        )
    with pytest.raises(PermissionError, match="training binding"):
        authorization.claim_for_training(
            model_config=authorization.model_config,
            cache=authorization.preflight.population.cache,
            schedule=authorization.preflight.schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
            device="cpu",
        )
    wrong = CoverageStatePACREConfig(
        feature_channels=2,
        feature_stride=2,
        width=5,
    )
    with pytest.raises(PermissionError, match="training binding"):
        run_pacre_pmope_bounded_400(
            authorization,
            wrong,
            run_id=PACRE_BOUNDED_RUN_ID,
            device="cuda:0",
        )
    assert called is False
    assert not authorization.consumed


def test_result_cannot_promote_an_arbitrary_all_true_decision(
    monkeypatch: pytest.MonkeyPatch,
    generated_graph,
    pacre_output_claim,
) -> None:
    authorization, _, _ = _authorization(
        monkeypatch,
        generated_graph,
        pacre_output_claim,
    )
    training = _training_bundle(authorization)
    diagnostic = _diagnostic(authorization, training.model)
    forged_decision = _decision(diagnostic)
    monkeypatch.setattr(
        bounded_runner,
        "_model_parameter_devices",
        lambda model: ("cuda:0",),
    )
    authorization.claim_for_training(
        model_config=authorization.model_config,
        cache=authorization.preflight.population.cache,
        schedule=authorization.preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
        device="cuda:0",
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        initial_model = build_pacre_training_model(
            authorization.model_config
        )
    authorization.consume_for_training(
        model=initial_model,
        model_config=authorization.model_config,
        cache=authorization.preflight.population.cache,
        schedule=authorization.preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
        device="cuda:0",
        objective=PACRE_PMOPE_OBJECTIVE,
        initial_model_fingerprint=(
            coverage_state_model_fingerprint(initial_model)
        ),
    )
    authorization.verify_for_run(
        cache=authorization.preflight.population.cache,
        schedule=authorization.preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
    )
    checks = bounded_runner._bounded_result_checks(
        authorization,
        training,
        diagnostic,
        forged_decision,
        run_id=PACRE_BOUNDED_RUN_ID,
        training_invocations=1,
        evaluation_invocations=1,
        decision_invocations=1,
    )
    result = CoverageStatePACREBoundedRunResult(
        run_id=PACRE_BOUNDED_RUN_ID,
        authorization=authorization,
        training=training,
        diagnostic=diagnostic,
        decision=forged_decision,
        training_invocations=1,
        zero_level_evaluation_invocations=1,
        decision_invocations=1,
        checks=checks,
    )
    assert dict(result.checks)["06_exact_pacre_decision"] is False
    assert not result.bounded_gate_passed
    assert not result.formal800_eligible


def test_concurrent_claims_have_one_winner_and_preserve_cpu_rng(
    monkeypatch: pytest.MonkeyPatch,
    generated_graph,
    pacre_output_claim,
) -> None:
    authorization, _, _ = _authorization(
        monkeypatch,
        generated_graph,
        pacre_output_claim,
    )
    copies = tuple(copy.copy(authorization) for _ in range(8))
    barrier = Barrier(len(copies))
    before = torch.random.get_rng_state().clone()

    def claim(value):
        barrier.wait()
        try:
            value.claim_for_training(
                model_config=value.model_config,
                cache=value.preflight.population.cache,
                schedule=value.preflight.schedule,
                scope=COVERAGE_STATE_BOUNDED_SCOPE,
                device="cuda:0",
            )
            return "claimed"
        except PermissionError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=len(copies)) as executor:
        outcomes = tuple(executor.map(claim, copies))

    assert outcomes.count("claimed") == 1
    assert outcomes.count("rejected") == len(copies) - 1
    assert torch.equal(before, torch.random.get_rng_state())
    assert authorization.reserved
    assert authorization.attempt_execution_ledger["claim_count"] == 1


def test_runner_reserves_before_deterministic_cuda_context_and_marks_failure(
    monkeypatch: pytest.MonkeyPatch,
    generated_graph,
    pacre_output_claim,
) -> None:
    authorization, _, _ = _authorization(
        monkeypatch,
        generated_graph,
        pacre_output_claim,
    )
    entered = False

    def deterministic(device):
        nonlocal entered
        assert device == torch.device("cuda:0")
        assert authorization.reserved
        entered = True
        return nullcontext()

    def fail_training(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("generated training failure")

    monkeypatch.setattr(
        bounded_runner,
        "_deterministic_execution",
        deterministic,
    )
    monkeypatch.setattr(
        bounded_runner,
        "train_pacre_pmope_candidate",
        fail_training,
    )
    with pytest.raises(RuntimeError, match="generated training failure"):
        run_pacre_pmope_bounded_400(
            authorization,
            authorization.model_config,
            run_id=PACRE_BOUNDED_RUN_ID,
            device="cuda:0",
        )

    assert entered
    ledger = authorization.attempt_execution_ledger
    assert ledger["state"] == "failed"
    assert ledger["claim_count"] == 1
    assert ledger["consume_count"] == 0
    assert ledger["failure_count"] == 1
    assert bounded_runner._is_sha256(
        ledger["training_binding_fingerprint"]
    )
    with pytest.raises(PermissionError, match="no longer available"):
        authorization.claim_for_training(
            model_config=authorization.model_config,
            cache=authorization.preflight.population.cache,
            schedule=authorization.preflight.schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
            device="cuda:0",
        )


def test_result_rejects_cpu_model_with_cuda_result_string(
    monkeypatch: pytest.MonkeyPatch,
    generated_graph,
    pacre_output_claim,
) -> None:
    authorization, _, _ = _authorization(
        monkeypatch,
        generated_graph,
        pacre_output_claim,
    )
    training = _training_bundle(authorization)
    diagnostic = _diagnostic(authorization, training.model)
    decision = _decision(diagnostic)
    authorization.claim_for_training(
        model_config=authorization.model_config,
        cache=authorization.preflight.population.cache,
        schedule=authorization.preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
        device="cuda:0",
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        initial_model = build_pacre_training_model(
            authorization.model_config
        )
    with monkeypatch.context() as context:
        context.setattr(
            bounded_runner,
            "_model_parameter_devices",
            lambda model: ("cuda:0",),
        )
        authorization.consume_for_training(
            model=initial_model,
            model_config=authorization.model_config,
            cache=authorization.preflight.population.cache,
            schedule=authorization.preflight.schedule,
            scope=COVERAGE_STATE_BOUNDED_SCOPE,
            device="cuda:0",
            objective=PACRE_PMOPE_OBJECTIVE,
            initial_model_fingerprint=(
                coverage_state_model_fingerprint(initial_model)
            ),
        )
    authorization.verify_for_run(
        cache=authorization.preflight.population.cache,
        schedule=authorization.preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
    )
    checks = dict(
        bounded_runner._bounded_result_checks(
            authorization,
            training,
            diagnostic,
            decision,
            run_id=PACRE_BOUNDED_RUN_ID,
            training_invocations=1,
            evaluation_invocations=1,
            decision_invocations=1,
        )
    )
    assert checks["03_seed42_10x40_compute_ledger"] is False
