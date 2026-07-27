from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
)
from cure_lite.experiment.coverage_state_pmope_dataset_free import (
    run_coverage_state_pmope_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_pmope_dr_gate import (
    COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED,
    COVERAGE_STATE_PMOPE_DR_EXPECTED_DATASET_FREE_FINGERPRINT,
    COVERAGE_STATE_PMOPE_DR_EXPECTED_POPULATION_FINGERPRINT,
    COVERAGE_STATE_PMOPE_DR_GATE_SCHEMA,
    CoverageStatePMOPEDRGateReceipt,
    load_coverage_state_pmope_v17_binding,
    run_coverage_state_pmope_dr_gate,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)


_ROOT = Path(__file__).resolve().parents[1]
_V17_INPUTS = (
    _ROOT
    / "runs/irstd1k_stage_a_seed42/"
    "cure_lite_cmif_v17_support_oriented_bounded_400_r1/"
    "receipts/inputs.json"
)


@pytest.fixture(scope="module")
def real_inputs():
    payload = json.loads(_V17_INPUTS.read_text(encoding="utf-8"))
    paths = payload["source_binding"]["paths"]
    return build_coverage_state_real_dr_inputs(
        manifest_path=_ROOT / paths["manifest"],
        state_index_path=_ROOT / paths["state_index"],
        geometry_config_path=_ROOT / paths["geometry_config"],
        geometry_receipt_path=_ROOT / paths["geometry_receipt"],
        observability_config_path=_ROOT / paths["observability_config"],
    )


@pytest.fixture(scope="module")
def bounded_population(real_inputs):
    return build_coverage_state_bounded_population(
        real_inputs.scalar_cache,
        seed=COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED,
    )


@pytest.fixture(scope="module")
def receipt(
    real_inputs,
    bounded_population,
) -> CoverageStatePMOPEDRGateReceipt:
    return run_coverage_state_pmope_dr_gate(
        dataset_free_receipt=(
            run_coverage_state_pmope_dataset_free_gate()
        ),
        real_inputs=real_inputs,
        bounded_population=bounded_population,
    )


def test_frozen_v17_result_and_source_closure_are_bound() -> None:
    first = load_coverage_state_pmope_v17_binding()
    replay = load_coverage_state_pmope_v17_binding()
    assert first.canonical_payload() == replay.canonical_payload()
    assert first.receipt_fingerprint == replay.receipt_fingerprint
    payload = first.canonical_payload()
    assert payload["complete_fingerprint"].startswith("50a9963a")
    assert len(payload["artifact_files"]) == 17
    assert len(payload["source_closure"]["source_members"]) == 40
    assert len(payload["controls"]) == 3
    assert all(payload["checks"].values())


def test_gate_binds_only_seed42_real_dr_and_passes(
    receipt: CoverageStatePMOPEDRGateReceipt,
) -> None:
    assert receipt.all_pass
    assert receipt.execution_seed == 42
    assert receipt.dataset_free_receipt_fingerprint == (
        COVERAGE_STATE_PMOPE_DR_EXPECTED_DATASET_FREE_FINGERPRINT
    )
    assert receipt.bounded_population_fingerprint == (
        COVERAGE_STATE_PMOPE_DR_EXPECTED_POPULATION_FINGERPRINT
    )
    assert all(dict(receipt.checks).values())
    payload = receipt.canonical_payload()
    assert payload["schema_version"] == COVERAGE_STATE_PMOPE_DR_GATE_SCHEMA
    assert payload["runtime_splits"] == ["D_R"]
    assert payload["execution_accounting"] == {
        "execution_seed": 42,
        "clean_pair_gradient_probes": 16,
        "optimizer_construction_count": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "calibration_performed": False,
        "inference_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    assert payload["bounded_400_authorized"] is False
    assert payload["formal_800_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False
    assert payload["claim_boundary"]["multi_seed_claim_supported"] is False


def test_all_clean_pairs_have_positive_added_support_measure(
    receipt: CoverageStatePMOPEDRGateReceipt,
) -> None:
    rows = receipt.geometry["mass_rows"]
    assert len(rows) == 16
    assert all(row["added_pixel_count"] > 0 for row in rows)
    assert all(
        float.fromhex(row["integration_mass_hex"]) > 0.0
        for row in rows
    )
    stats = receipt.geometry["mass_statistics"]
    assert stats["count"] == 16
    assert float.fromhex(stats["minimum_hex"]) == pytest.approx(
        0.0476190485060215
    )
    assert float.fromhex(stats["median_hex"]) == pytest.approx(
        0.2847222238779068
    )
    assert float.fromhex(stats["mean_hex"]) == pytest.approx(
        0.2716567537281662
    )
    assert float.fromhex(stats["maximum_hex"]) == pytest.approx(
        0.3333333432674408
    )
    assert stats["threshold_derived_from_observed_values"] is False
    assert stats["required_contract"] == (
        "strictly_positive_per_clean_pair"
    )


def test_target_and_null_geometry_contracts_hold(
    receipt: CoverageStatePMOPEDRGateReceipt,
) -> None:
    geometry = receipt.geometry
    assert geometry["target_fields_strictly_nonzero_on_valid"] is True
    assert geometry["valid_target_field_pixel_count"] > 0
    assert geometry["pair_counts"] == {
        "clean_positive": 16,
        "component_null": 16,
        "identity_null": 16,
        "diagnostic_only_component_null": 1,
    }
    assert geometry["component_geometry_contract"] is True
    assert geometry["identity_geometry_contract"] is True
    assert geometry["diagnostic_component_geometry_contract"] is True


def test_every_clean_pair_has_positive_loss_and_scalar_gradient(
    receipt: CoverageStatePMOPEDRGateReceipt,
) -> None:
    rows = receipt.gradient_rows
    assert len(rows) == 16
    assert all(float.fromhex(row["loss_hex"]) > 0.0 for row in rows)
    assert all(
        float.fromhex(row["scalar_energy_gradient_l2_hex"]) > 0.0
        for row in rows
    )
    assert all(
        row["scalar_energy_gradient_finite"] is True for row in rows
    )
    assert all(
        row["scalar_energy_gradient_nonzero_count"] == 32 for row in rows
    )
    assert receipt.initial_model_fingerprint == (
        receipt.final_model_fingerprint
    )
    assert receipt.input_before_fingerprint == (
        receipt.input_after_fingerprint
    )
    assert receipt.parameter_grad_buffers_unretained is True
    assert receipt.global_cpu_rng_preserved is True


def test_same_seed42_replay_is_exact_not_a_multiseed_claim(
    receipt: CoverageStatePMOPEDRGateReceipt,
) -> None:
    first_payload = receipt.canonical_payload()
    first_fingerprint = receipt.receipt_fingerprint
    receipt.verify_unchanged()
    assert receipt.execution_seed == 42
    assert receipt.canonical_payload() == first_payload
    assert receipt.receipt_fingerprint == first_fingerprint
    assert (
        first_payload["claim_boundary"][
            "same_seed_deterministic_replay_only"
        ]
        is True
    )


def test_receipt_tampering_fails_closed(
    receipt: CoverageStatePMOPEDRGateReceipt,
) -> None:
    with pytest.raises(RuntimeError, match="gate evidence changed"):
        replace(receipt, execution_seed=43)
