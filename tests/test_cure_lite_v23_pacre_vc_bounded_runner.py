from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from threading import Barrier

import pytest

import cure_lite_v23.bounded_runner as bounded_runner
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
)
from cure_lite_v23.decision import PACRE_VC_BOUNDED_RUN_ID
from cure_lite_v23.pacre_vc import (
    PACRE_VC_CANDIDATE,
    CoverageStatePACREVerifierCorrectedConfig,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)


class _GeneratedRealInputs(CoverageStateRealDRInputs):
    def __init__(self, scalar_cache) -> None:
        object.__setattr__(
            self,
            "source_binding",
            type("_Binding", (), {"split": "D_R"})(),
        )
        object.__setattr__(self, "scalar_cache", scalar_cache)
        object.__setattr__(
            self,
            "build_fingerprint",
            stable_fingerprint(
                {"generated_real_inputs": scalar_cache.cache_fingerprint}
            ),
        )

    def verify_unchanged(self) -> None:
        self.scalar_cache.verify_unchanged()
        if self.scalar_cache.raw_catalog.split != "D_R":
            raise RuntimeError("generated input split changed")


class _Receipt:
    def __init__(
        self,
        *,
        dataset_fingerprint: str,
        real_inputs,
        population,
        passed: bool,
    ) -> None:
        self.dataset_free_receipt_fingerprint = dataset_fingerprint
        self.real_inputs_fingerprint = real_inputs.build_fingerprint
        self.population_fingerprint = population.population_fingerprint
        self.cache_fingerprint = population.bounded_cache_fingerprint
        self.gate_passed = passed
        self.decision = (
            bounded_runner.PACRE_DR_PASS_DECISION
            if passed
            else "PACRE_V23_D_R_STRUCTURAL_FAIL"
        )
        self.probe: dict[str, object] = {}
        self._mapping_mutation: dict[str, object] = {}

    def bind_probe(self, model_config) -> None:
        (
            contract_json,
            contract_fingerprint,
            initial_fingerprint,
            model_fqcn,
            _,
        ) = bounded_runner._model_binding(model_config)
        self.probe = {
            "execution_seed": 42,
            "model_fqcn": model_fqcn,
            "config_fqcn": bounded_runner.PACRE_CONFIG_FQCN,
            "model_contract": json.loads(contract_json),
            "model_contract_fingerprint": contract_fingerprint,
            "initial_model_fingerprint": initial_fingerprint,
            "final_model_fingerprint": initial_fingerprint,
            "training_performed": False,
            "D_R_accessed": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "generated-v23-dr-receipt-v1",
            "candidate": PACRE_VC_CANDIDATE,
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "real_inputs_fingerprint": self.real_inputs_fingerprint,
            "population_fingerprint": self.population_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "probe": self.probe,
            "gate_passed": self.gate_passed,
            "decision": self.decision,
            "training_performed": False,
            "D_R_accessed": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            **self._mapping_mutation,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(
        self,
        *,
        dataset_free_receipt,
        real_inputs,
        bounded_population,
    ) -> None:
        assert (
            dataset_free_receipt["receipt_fingerprint"]
            == self.dataset_free_receipt_fingerprint
        )
        assert real_inputs.build_fingerprint == self.real_inputs_fingerprint
        assert (
            bounded_population.population_fingerprint
            == self.population_fingerprint
        )


@pytest.fixture(autouse=True)
def _isolated_attempt_registry():
    with bounded_runner._ATTEMPT_REGISTRY_LOCK:
        bounded_runner._ATTEMPT_REGISTRY.clear()
    yield
    with bounded_runner._ATTEMPT_REGISTRY_LOCK:
        bounded_runner._ATTEMPT_REGISTRY.clear()


@pytest.fixture
def graph(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    source_cache = make_bounded_training_scalar_cache()
    population = build_coverage_state_bounded_population(source_cache)
    preflight = prepare_coverage_state_bounded_preflight(population)
    real_inputs = _GeneratedRealInputs(source_cache)
    model_config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    dataset_body: dict[str, object] = {
        "schema_version": "generated-v23-dataset-free-v1",
        "candidate": PACRE_VC_CANDIDATE,
        "parameter_count": model_config.expected_parameter_count,
        "gate_passed": True,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    dataset = {
        **dataset_body,
        "receipt_fingerprint": stable_fingerprint(dataset_body),
    }

    def validate_dataset(payload):
        body = dict(payload)
        fingerprint = body.pop("receipt_fingerprint")
        if (
            fingerprint != stable_fingerprint(body)
            or body.get("candidate") != PACRE_VC_CANDIDATE
            or body.get("gate_passed") is not True
        ):
            raise ValueError("invalid generated dataset-free receipt")
        return fingerprint

    monkeypatch.setattr(
        bounded_runner,
        "_validate_dataset_free_receipt",
        validate_dataset,
    )
    monkeypatch.setattr(
        bounded_runner,
        "CoverageStatePACREDRGateReceipt",
        _Receipt,
    )

    output = tmp_path / "fixed-v23-bounded-output"
    monkeypatch.setattr(
        bounded_runner,
        "PACRE_BOUNDED_OUTPUT_PATH",
        output,
    )
    body = {
        "schema_version": (
            bounded_runner.PACRE_BOUNDED_ATTEMPT_RECEIPT_SCHEMA
        ),
        "run_id": PACRE_VC_BOUNDED_RUN_ID,
        "output_repo_path": (
            bounded_runner.PACRE_BOUNDED_OUTPUT_REPO_PATH
        ),
        "config_fingerprint": "a" * 64,
        "runtime": {
            "device": "cuda:0",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "temperature_wrapper_repo_path": (
                bounded_runner
                .PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH
            ),
            "temperature_wrapper_file_sha256": (
                bounded_runner
                .PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256
            ),
            "pause_temperature_c": 82,
            "resume_temperature_c": 75,
        },
        "candidate": PACRE_VC_CANDIDATE,
        "objective": bounded_runner.PACRE_PMOPE_OBJECTIVE,
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
            dataset["receipt_fingerprint"]
        ),
        "dataset_free_invocations_before_claim": 1,
        "single_attempt": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "formal_800_authorized": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    attempt = {**body, "receipt_fingerprint": stable_fingerprint(body)}
    output.mkdir()
    (output / "attempt.json").write_text(
        canonical_json(attempt) + "\n",
        encoding="utf-8",
    )
    (output / ".incomplete").touch()
    claim = bounded_runner.load_pacre_bounded_output_claim()
    return preflight, real_inputs, model_config, dataset, claim


def _receipt(graph, *, passed: bool) -> _Receipt:
    preflight, real_inputs, model_config, dataset, _ = graph
    receipt = _Receipt(
        dataset_fingerprint=dataset["receipt_fingerprint"],
        real_inputs=real_inputs,
        population=preflight.population,
        passed=passed,
    )
    receipt.bind_probe(model_config)
    return receipt


def test_authorization_binds_v23_dr_mapping_fingerprint_and_budget(
    graph,
) -> None:
    preflight, real_inputs, model_config, dataset, claim = graph
    receipt = _receipt(graph, passed=True)
    authorization = (
        bounded_runner.prepare_pacre_vc_bounded_run_authorization(
            preflight,
            dataset,
            receipt,
            real_inputs,
            model_config,
            output_claim=claim,
            run_id=PACRE_VC_BOUNDED_RUN_ID,
        )
    )
    payload = authorization.canonical_payload()

    assert authorization.available
    assert payload["D_R_gate"] == receipt.canonical_payload()
    assert payload["D_R_gate_receipt_fingerprint"] == (
        receipt.receipt_fingerprint
    )
    assert payload["population_fingerprint"] == (
        preflight.population.population_fingerprint
    )
    assert payload["budget"] == {
        "seed": 42,
        "epochs": 10,
        "steps_per_epoch": 40,
        "updates": 400,
        "objectives": 1,
    }
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["formal_800_authorized"] is False
    assert payload["formal800_status"] == (
        "BLOCKED_PENDING_SEPARATE_PREREGISTRATION"
    )


def test_dr_fail_and_mapping_fingerprint_drift_fail_closed_before_model(
    graph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, real_inputs, model_config, dataset, claim = graph
    failed = _receipt(graph, passed=False)
    passed = _receipt(graph, passed=True)
    reached_model_binding = False
    real_model_binding = bounded_runner._model_binding

    def model_spy(config):
        nonlocal reached_model_binding
        reached_model_binding = True
        return real_model_binding(config)

    monkeypatch.setattr(bounded_runner, "_model_binding", model_spy)
    with pytest.raises(PermissionError, match="structural PASS"):
        bounded_runner.prepare_pacre_vc_bounded_run_authorization(
            preflight,
            dataset,
            failed,
            real_inputs,
            model_config,
            output_claim=claim,
            run_id=PACRE_VC_BOUNDED_RUN_ID,
        )
    assert reached_model_binding is False

    frozen_fingerprint = passed.receipt_fingerprint
    passed._mapping_mutation["unexpected_after_fingerprint"] = True
    # Simulate a stored receipt fingerprint that no longer matches its map.
    monkeypatch.setattr(
        type(passed),
        "receipt_fingerprint",
        property(lambda self: frozen_fingerprint),
    )
    with pytest.raises(PermissionError, match="structural PASS"):
        bounded_runner.prepare_pacre_vc_bounded_run_authorization(
            preflight,
            dataset,
            passed,
            real_inputs,
            model_config,
            output_claim=claim,
            run_id=PACRE_VC_BOUNDED_RUN_ID,
        )


def test_single_use_claim_has_exactly_one_concurrent_winner(graph) -> None:
    preflight, real_inputs, model_config, dataset, claim = graph
    authorization = (
        bounded_runner.prepare_pacre_vc_bounded_run_authorization(
            preflight,
            dataset,
            _receipt(graph, passed=True),
            real_inputs,
            model_config,
            output_claim=claim,
            run_id=PACRE_VC_BOUNDED_RUN_ID,
        )
    )
    copies = tuple(copy.copy(authorization) for _ in range(6))
    barrier = Barrier(len(copies))

    def claim_one(value):
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
        except PermissionError as error:
            return f"rejected:{error}"

    with ThreadPoolExecutor(max_workers=len(copies)) as executor:
        outcomes = tuple(executor.map(claim_one, copies))

    assert outcomes.count("claimed") == 1
    assert sum(
        value.startswith("rejected:") for value in outcomes
    ) == len(copies) - 1
    assert authorization.reserved
    assert authorization.attempt_execution_ledger["claim_count"] == 1
