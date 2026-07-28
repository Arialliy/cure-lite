from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from cure_lite.experiment.coverage_state_uscope_dataset_free import (
    COVERAGE_STATE_USCOPE_IMPLEMENTATION_PATHS,
    COVERAGE_STATE_USCOPE_MARGIN,
    CoverageStateUSCOPEDatasetFreeReceipt,
    recompute_coverage_state_uscope_dataset_free_checks,
    run_coverage_state_uscope_dataset_free_gate,
)


@pytest.fixture(scope="module")
def receipt() -> CoverageStateUSCOPEDatasetFreeReceipt:
    return run_coverage_state_uscope_dataset_free_gate()


def test_uscope_generated_gate_passes(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    assert receipt.all_pass
    assert all(dict(receipt.checks).values())
    assert COVERAGE_STATE_USCOPE_MARGIN == 0.225
    assert receipt.canonical_payload()["objective_policy"] == (
        "uniform_sobolev_chebyshev_orthant_projection_energy_v1"
    )


def test_exact_linf_product_power_and_statewise_batch_mean(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.exact_product_probe
    assert probe["state_count"] == 2
    assert probe["gamma_exact"] is True
    assert probe["sobolev_power_exact"] is True
    assert probe["chebyshev_power_exact"] is True
    assert probe["product_power_exact"] is True
    assert probe["per_state_loss_exact"] is True
    assert probe["batch_mean_exact"] is True
    assert len(probe["gamma_hex"]) == 2
    assert len(probe["per_state_loss_hex"]) == 2


def test_gamma_below_margin_is_a_strict_sign_certificate(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.gamma_certificate_probe
    assert probe["certified_gamma_strictly_below_margin"] is True
    assert probe["certified_plus_sign_exact"] is True
    assert probe["certified_minus_sign_exact"] is True
    assert probe["boundary_gamma_at_least_margin"] is True
    assert probe["boundary_minus_sign_not_exact"] is True
    assert probe["certificate_is_sufficient_not_necessary"] is True


def test_one_bad_pixel_is_not_spatially_diluted(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.single_pixel_probe
    assert probe["gamma_size_invariant"] is True
    assert probe["one_bad_pixel_not_diluted"] is True
    assert probe["small"]["pixel_count"] == 16 * 16
    assert probe["full"]["pixel_count"] == 256 * 256
    assert probe["small"]["gamma_hex"] == probe["full"]["gamma_hex"]
    assert probe["small"][
        "bad_pixel_descent_moves_background_positive"
    ] is True
    assert probe["full"][
        "bad_pixel_descent_moves_background_positive"
    ] is True
    assert probe["target_gamma_size_invariant"] is True
    assert probe["one_bad_target_pixel_not_diluted"] is True
    assert probe["target_small"]["pixel_count"] == 16 * 16
    assert probe["target_full"]["pixel_count"] == 256 * 256
    assert (
        probe["target_small"]["gamma_hex"]
        == probe["target_full"]["gamma_hex"]
    )
    assert probe["target_small"][
        "bad_pixel_descent_has_expected_direction"
    ] is True
    assert probe["target_full"][
        "bad_pixel_descent_has_expected_direction"
    ] is True


def test_target_and_background_descent_directions_are_correct(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.gradient_direction_probe
    assert probe["target_gradient_positive"] is True
    assert probe["target_descent_negative"] is True
    assert probe["background_gradient_negative"] is True
    assert probe["background_descent_positive"] is True
    assert probe["target_gradient_finite"] is True
    assert probe["background_gradient_finite"] is True


def test_occupied_hidden_negative_is_captured_on_full_valid_domain(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.occupied_hidden_negative_probe
    assert probe["occupied_pixel"] is True
    assert probe["raw_hidden_negative"] is True
    assert probe["completion_masks_hidden_negative"] is True
    assert probe["valid_domain_includes_occupied_pixel"] is True
    assert probe["full_domain_violation_positive"] is True
    assert probe["gamma_positive"] is True
    assert probe["gamma_above_margin"] is True
    assert probe["loss_positive"] is True
    assert probe["descent_moves_field_positive"] is True


def test_outside_valid_domain_is_explicitly_excluded(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.invalid_domain_probe
    assert probe["probe_pixel_outside_valid_domain"] is True
    assert probe["outside_plus_violation_exact_zero"] is True
    assert probe["outside_minus_violation_exact_zero"] is True
    assert probe["gamma_unchanged"] is True
    assert probe["loss_unchanged"] is True
    assert probe["outside_domain_has_no_certificate"] is True


def test_uscope_and_pmope_share_q_and_zero_set_but_not_finite_risk(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.zero_set_and_finite_risk_probe
    assert probe["zero_violation_plus_same"] is True
    assert probe["zero_violation_minus_same"] is True
    assert probe["zero_uscope_loss_exact"] is True
    assert probe["zero_pmope_loss_exact"] is True
    assert probe["zero_gamma_exact"] is True
    assert probe["finite_violation_plus_same"] is True
    assert probe["finite_violation_minus_same"] is True
    assert probe["finite_uscope_positive"] is True
    assert probe["finite_pmope_positive"] is True
    assert probe["finite_risks_different"] is True
    assert probe["finite_uscope_greater_than_pmope"] is True


def test_same_sign_response_is_diagnostic_only(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.same_sign_response_diagnostic_probe
    assert probe["same_sign_response_pixel_count"] > 0
    assert probe["same_sign_response_has_error"] is True
    assert probe["endpoint_violations_exact_zero"] is True
    assert probe["gamma_exact_zero"] is True
    assert probe["uscope_loss_exact_zero"] is True
    assert probe["response_consumed_by_objective"] is False
    assert probe["response_is_diagnostic_only"] is True


def test_receipt_recomputes_and_replays_exactly(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    recomputed = recompute_coverage_state_uscope_dataset_free_checks(
        implementation_binding=receipt.implementation_binding,
        generated_replay_fingerprint=(
            receipt.generated_replay_fingerprint
        ),
        exact_product_probe=receipt.exact_product_probe,
        gamma_certificate_probe=receipt.gamma_certificate_probe,
        single_pixel_probe=receipt.single_pixel_probe,
        gradient_direction_probe=receipt.gradient_direction_probe,
        occupied_hidden_negative_probe=(
            receipt.occupied_hidden_negative_probe
        ),
        invalid_domain_probe=receipt.invalid_domain_probe,
        zero_set_and_finite_risk_probe=(
            receipt.zero_set_and_finite_risk_probe
        ),
        same_sign_response_diagnostic_probe=(
            receipt.same_sign_response_diagnostic_probe
        ),
    )
    assert recomputed == receipt.checks
    replay = run_coverage_state_uscope_dataset_free_gate()
    assert replay.canonical_payload() == receipt.canonical_payload()
    assert replay.receipt_fingerprint == receipt.receipt_fingerprint


def test_receipt_has_no_runtime_data_or_training_authority(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    payload = receipt.canonical_payload()
    assert payload["runtime_splits"] == []
    assert payload["D_R_accessed"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["dataset_training_performed"] is False
    assert payload["optimizer_constructed"] is False
    assert payload["optimizer_steps"] == 0
    assert payload["bounded_400_authorized"] is False
    assert payload["formal_800_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False


def test_implementation_binding_is_complete(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    assert tuple(
        path for path, _ in receipt.implementation_binding
    ) == COVERAGE_STATE_USCOPE_IMPLEMENTATION_PATHS
    assert all(
        len(digest) == 64
        for _, digest in receipt.implementation_binding
    )


def test_receipt_rejects_nested_mutation() -> None:
    receipt = run_coverage_state_uscope_dataset_free_gate()
    receipt.exact_product_probe["gamma_exact"] = False
    with pytest.raises(RuntimeError, match="evidence changed"):
        _ = receipt.all_pass
    with pytest.raises(RuntimeError, match="evidence changed"):
        _ = receipt.receipt_fingerprint


def test_receipt_rejects_tampered_checks(
    receipt: CoverageStateUSCOPEDatasetFreeReceipt,
) -> None:
    tampered = list(receipt.checks)
    name, value = tampered[0]
    tampered[0] = (name, not value)
    with pytest.raises(RuntimeError, match="evidence changed"):
        CoverageStateUSCOPEDatasetFreeReceipt(
            implementation_binding=receipt.implementation_binding,
            exact_product_probe=deepcopy(receipt.exact_product_probe),
            gamma_certificate_probe=deepcopy(
                receipt.gamma_certificate_probe
            ),
            single_pixel_probe=deepcopy(receipt.single_pixel_probe),
            gradient_direction_probe=deepcopy(
                receipt.gradient_direction_probe
            ),
            occupied_hidden_negative_probe=deepcopy(
                receipt.occupied_hidden_negative_probe
            ),
            invalid_domain_probe=deepcopy(
                receipt.invalid_domain_probe
            ),
            zero_set_and_finite_risk_probe=deepcopy(
                receipt.zero_set_and_finite_risk_probe
            ),
            same_sign_response_diagnostic_probe=deepcopy(
                receipt.same_sign_response_diagnostic_probe
            ),
            generated_replay_fingerprint=(
                receipt.generated_replay_fingerprint
            ),
            checks=tuple(tampered),
            evidence_fingerprint=receipt.evidence_fingerprint,
        )


def test_gate_restores_callers_cpu_rng_state() -> None:
    torch.manual_seed(1919)
    before = torch.random.get_rng_state().clone()
    run_coverage_state_uscope_dataset_free_gate()
    assert torch.equal(before, torch.random.get_rng_state())
