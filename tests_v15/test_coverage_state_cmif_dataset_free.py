from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from cure_lite.experiment.coverage_state_cmif_dataset_free import (
    COVERAGE_STATE_CMIF_DATASET_FREE_SEEDS,
    COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT,
    CoverageStateCMIFDatasetFreeReceipt,
    recompute_coverage_state_cmif_dataset_free_checks,
    run_coverage_state_cmif_dataset_free_gate,
)


@pytest.fixture(scope="module")
def receipt() -> CoverageStateCMIFDatasetFreeReceipt:
    return run_coverage_state_cmif_dataset_free_gate()


def test_cmif_dataset_free_gate_passes_every_frozen_check(
    receipt: CoverageStateCMIFDatasetFreeReceipt,
) -> None:
    assert receipt.all_pass
    assert receipt.formal_config_payload[
        "expected_parameter_count"
    ] == COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
    assert receipt.formal_config_payload[
        "actual_parameter_count"
    ] == COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
    assert tuple(
        value["seed"] for value in receipt.reference_probes
    ) == COVERAGE_STATE_CMIF_DATASET_FREE_SEEDS
    assert all(dict(receipt.checks).values())


def test_cmif_dataset_free_receipt_recomputes_and_replays_exactly(
    receipt: CoverageStateCMIFDatasetFreeReceipt,
) -> None:
    recomputed = recompute_coverage_state_cmif_dataset_free_checks(
        formal_config_payload=receipt.formal_config_payload,
        parameter_names=receipt.parameter_names,
        parameter_contract=receipt.parameter_contract,
        implementation_binding=receipt.implementation_binding,
        reference_probes=receipt.reference_probes,
        phase_probe=receipt.phase_probe,
        center_probe=receipt.center_probe,
        null_probe=receipt.null_probe,
        gauge_probe=receipt.gauge_probe,
        endpoint_probe=receipt.endpoint_probe,
        locality_probe=receipt.locality_probe,
        gradient_probe=receipt.gradient_probe,
    )
    assert recomputed == receipt.checks
    replay = run_coverage_state_cmif_dataset_free_gate()
    assert replay.canonical_payload() == receipt.canonical_payload()
    assert replay.receipt_fingerprint == receipt.receipt_fingerprint


def test_dataset_free_receipt_contains_no_dataset_or_training_authority(
    receipt: CoverageStateCMIFDatasetFreeReceipt,
) -> None:
    payload = receipt.canonical_payload()
    assert payload["runtime_splits"] == []
    assert payload["D_R_accessed"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["dataset_training_performed"] is False
    assert payload["synthetic_gradient_probe_optimizer_steps"] == 1
    assert payload["bounded_400_authorized"] is False
    assert payload["formal_800_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False


def test_receipt_rejects_a_tampered_gate(
    receipt: CoverageStateCMIFDatasetFreeReceipt,
) -> None:
    tampered = list(receipt.checks)
    name, value = tampered[0]
    tampered[0] = (name, not value)
    with pytest.raises(RuntimeError, match="evidence changed"):
        CoverageStateCMIFDatasetFreeReceipt(
            formal_config_payload=deepcopy(
                receipt.formal_config_payload
            ),
            parameter_names=receipt.parameter_names,
            parameter_contract=receipt.parameter_contract,
            implementation_binding=receipt.implementation_binding,
            reference_probes=deepcopy(receipt.reference_probes),
            phase_probe=deepcopy(receipt.phase_probe),
            center_probe=deepcopy(receipt.center_probe),
            null_probe=deepcopy(receipt.null_probe),
            gauge_probe=deepcopy(receipt.gauge_probe),
            endpoint_probe=deepcopy(receipt.endpoint_probe),
            locality_probe=deepcopy(receipt.locality_probe),
            gradient_probe=deepcopy(receipt.gradient_probe),
            checks=tuple(tampered),
            evidence_fingerprint=receipt.evidence_fingerprint,
        )


def test_receipt_detects_post_construction_nested_mutation() -> None:
    receipt = run_coverage_state_cmif_dataset_free_gate()
    original_fingerprint = receipt.receipt_fingerprint
    receipt.formal_config_payload["actual_parameter_count"] = 1
    with pytest.raises(RuntimeError, match="evidence changed"):
        _ = receipt.all_pass
    with pytest.raises(RuntimeError, match="evidence changed"):
        _ = receipt.receipt_fingerprint
    assert original_fingerprint


def test_dataset_free_gate_restores_the_callers_cpu_rng_state() -> None:
    torch.manual_seed(1917)
    before = torch.random.get_rng_state().clone()
    run_coverage_state_cmif_dataset_free_gate()
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
