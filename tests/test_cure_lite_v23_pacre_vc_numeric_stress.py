from __future__ import annotations

import json

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite_v23.numeric_stress import (
    PACRE_VC_FORMAL_STRESS_SEEDS,
    generate_pacre_vc_formal_input,
    run_pacre_vc_formal_numeric_stress_receipt,
    run_pacre_vc_formal_numeric_stress_run,
    run_pacre_vc_scalar_counterexample_receipt,
)


def _assert_self_fingerprint(
    payload: dict[str, object],
    field: str,
) -> None:
    body = dict(payload)
    observed = body.pop(field)
    assert observed == stable_fingerprint(body)


def test_formal_inputs_use_independent_cpu_generator_and_frozen_shape() -> None:
    before = torch.random.get_rng_state().clone()
    first = generate_pacre_vc_formal_input(42)
    second = generate_pacre_vc_formal_input(42)

    assert torch.equal(before, torch.random.get_rng_state())
    assert first.feature.device == torch.device("cpu")
    assert first.occupancy.device == torch.device("cpu")
    assert first.feature.shape == (1, 64, 64, 64)
    assert first.occupancy.shape == (1, 1, 256, 256)
    assert first.feature.dtype == torch.float32
    assert first.occupancy.dtype == torch.bool
    assert first.manifest == second.manifest
    assert first.manifest["generator_device"] == "cpu"
    assert first.manifest["global_cpu_rng_preserved"] is True
    _assert_self_fingerprint(first.manifest, "manifest_fingerprint")


def test_two_frozen_scalar_cancellation_counterexamples() -> None:
    receipt = run_pacre_vc_scalar_counterexample_receipt()

    assert receipt["gate_passed"] is True
    assert len(receipt["cases"]) == 2
    assert [
        (case["common_hex"], case["residual_hex"])
        for case in receipt["cases"]
    ] == [
        (float(torch.tensor(100.0)).hex(), float(torch.tensor(0.1)).hex()),
        (float(torch.tensor(10.0)).hex(), float(torch.tensor(0.01)).hex()),
    ]
    assert all(
        case["legacy_v22_allclose"] is False
        and case["legacy_failure_reproduced"] is True
        and case["same_operand_exact_forward_replay"] is True
        and case["phase_semantics"]["passed"] is True
        for case in receipt["cases"]
    )
    assert receipt["legacy_formula_gate_eligible"] is False
    assert receipt["legacy_formula_decision_weight"] == 0
    _assert_self_fingerprint(receipt, "receipt_fingerprint")


def test_formal_cpu_seed_has_complete_honest_evidence() -> None:
    run = run_pacre_vc_formal_numeric_stress_run(
        device="cpu",
        seed=42,
    )

    assert run["gate_passed"] is True
    assert run["device_copy_raw_bits_preserved"] is True
    assert run["initial_model_state"]["passed"] is True
    assert run["all_fields_raw_parity"]["passed"] is True
    assert run["algebra_verification"]["passed"] is True
    assert run["algebra_verification"]["exact_replay"]["passed"] is True
    assert run["algebra_verification"]["phase_semantics"]["passed"] is True
    assert run["fp64_oracle"]["integrity"]["passed"] is True
    assert run["signal_swallow_ledger_integrity"]["passed"] is True
    assert run["fixed_readout_nonzero_observation"]["passed"] is True
    assert run["legacy_subtraction_diagnostics"]["gate_eligible"] is False
    assert run["legacy_six_check_replay"]["all_diagnostic_only"] is True
    assert run["ledger"] == {
        "complete": True,
        "field_tensor_count": 21,
        "algebra_replay_check_count": 15,
        "zero_bound_exact_check_count": 13,
        "bounded_hidden_check_count": 2,
        "phase_semantic_check_count": 2,
        "legacy_formula_count": 6,
        "swallow_observation_count": 10,
    }
    assert run["dataset_accessed"] is False
    assert run["cache_accessed"] is False
    assert run["D_R_accessed"] is False
    assert run["D_V_accessed"] is False
    assert run["D_T_accessed"] is False
    assert run["optimizer_constructed"] is False
    assert run["training_performed"] is False
    _assert_self_fingerprint(run, "run_fingerprint")


def test_formal_receipt_contains_only_summaries_for_all_seeds() -> None:
    receipt = run_pacre_vc_formal_numeric_stress_receipt(device="cpu")

    assert receipt["gate_passed"] is True
    assert receipt["required_seeds"] == list(
        PACRE_VC_FORMAL_STRESS_SEEDS
    )
    assert [run["seed"] for run in receipt["runs"]] == list(
        PACRE_VC_FORMAL_STRESS_SEEDS
    )
    assert all(run["gate_passed"] is True for run in receipt["runs"])
    assert (
        receipt["runtime_environment_verified_pre_input_generation"]
        is True
    )
    assert receipt["runtime_environment_verified_post_execution"] is True
    assert receipt["source_closure_verified_pre_input_generation"] is True
    assert receipt["source_closure_verified_post_execution"] is True
    # A tensor dump would be tens or hundreds of MB.  The canonical receipt
    # remains a compact ledger of hashes, shapes, counts, and scalar maxima.
    encoded = json.dumps(receipt, separators=(",", ":"))
    assert len(encoded) < 2_000_000
    _assert_self_fingerprint(receipt, "receipt_fingerprint")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="formal selected-device stress requires CUDA",
)
def test_formal_cuda0_receipt_passes_frozen_seed_matrix() -> None:
    receipt = run_pacre_vc_formal_numeric_stress_receipt(
        device="cuda:0",
    )

    assert receipt["device"] == "cuda:0"
    assert receipt["gate_passed"] is True
    assert [run["seed"] for run in receipt["runs"]] == [42, 43, 44]
    assert all(
        run["device_copy_raw_bits_preserved"] is True
        and run["all_fields_raw_parity"]["passed"] is True
        and run["algebra_verification"]["passed"] is True
        and run["fp64_oracle"]["integrity"]["passed"] is True
        for run in receipt["runs"]
    )
    _assert_self_fingerprint(receipt, "receipt_fingerprint")
