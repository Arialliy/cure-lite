from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from cure_lite.experiment.coverage_state_pmope_dataset_free import (
    COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_PMOPE_IMPLEMENTATION_PATHS,
    COVERAGE_STATE_PMOPE_MARGIN,
    COVERAGE_STATE_PMOPE_POLICY,
    CoverageStatePMOPEDatasetFreeReceipt,
    recompute_coverage_state_pmope_dataset_free_checks,
    run_coverage_state_pmope_dataset_free_gate,
)


@pytest.fixture(scope="module")
def receipt() -> CoverageStatePMOPEDatasetFreeReceipt:
    return run_coverage_state_pmope_dataset_free_gate()


def test_pmope_dataset_free_gate_passes_all_frozen_checks(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    assert receipt.all_pass
    assert all(dict(receipt.checks).values())
    assert COVERAGE_STATE_PMOPE_MARGIN == 0.225
    assert receipt.canonical_payload()["objective_policy"] == (
        COVERAGE_STATE_PMOPE_POLICY
    )
    assert receipt.formal_config_payload[
        "actual_parameter_count"
    ] == COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT


def test_gate_recomputes_and_independent_replays_are_exact(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    recomputed = recompute_coverage_state_pmope_dataset_free_checks(
        formal_config_payload=receipt.formal_config_payload,
        parameter_names=receipt.parameter_names,
        parameter_contract=receipt.parameter_contract,
        implementation_binding=receipt.implementation_binding,
        clean_probe=receipt.clean_probe,
        component_null_probe=receipt.component_null_probe,
        zero_semantics_probe=receipt.zero_semantics_probe,
        non_equivalence_probe=receipt.non_equivalence_probe,
        cmif_parameter_gradient_probe=(
            receipt.cmif_parameter_gradient_probe
        ),
        generated_replay_fingerprint=(
            receipt.generated_replay_fingerprint
        ),
    )
    assert recomputed == receipt.checks
    replay = run_coverage_state_pmope_dataset_free_gate()
    assert replay.canonical_payload() == receipt.canonical_payload()
    assert replay.receipt_fingerprint == receipt.receipt_fingerprint


def test_clean_pair_has_only_negative_minus_descent(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.clean_probe
    assert probe["plus_violation_exact_zero"] is True
    assert probe["plus_gradient_exact_zero"] is True
    assert probe["minus_violation_positive_on_added"] is True
    assert probe["minus_gradient_positive_on_added"] is True
    assert probe["minus_descent_direction_negative_on_added"] is True
    assert probe["minus_gradient_exact_zero_outside"] is True


def test_component_null_has_only_positive_minus_descent(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.component_null_probe
    assert probe["target_plus_is_all_exterior"] is True
    assert probe["target_minus_is_all_exterior"] is True
    assert probe["plus_gradient_exact_zero"] is True
    assert probe["minus_gradient_negative_on_removed"] is True
    assert probe["minus_descent_direction_positive_on_removed"] is True
    assert probe["minus_gradient_exact_zero_outside"] is True


def test_zero_loss_implies_raw_sign_and_completion_equivalence(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.zero_semantics_probe
    assert probe["zero_loss_exact"] is True
    assert probe["full_valid_domain_used"] is True
    assert probe["raw_plus_sign_set_exact"] is True
    assert probe["raw_minus_sign_set_exact"] is True
    assert probe["completion_plus_exact"] is True
    assert probe["completion_minus_exact"] is True
    assert probe["one_valid_zero_margin_violation_positive"] is True


def test_pmope_has_a_direct_non_equivalence_witness(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.non_equivalence_probe
    assert probe["deep_correct_sign_field"] is True
    assert probe["pmope_loss_exact_zero"] is True
    assert probe["identity_loss_positive"] is True
    assert probe["one_sided_feasible_cone_witness"] is True
    assert probe["sorr_loss_positive"] is True
    assert probe["separable_loss_positive"] is True
    assert probe["omco_loss_positive"] is True
    assert probe["all_four_old_objective_losses_positive"] is True
    assert probe["omco_coordinate_policy"] == (
        "fixed_orthogonal_sum_difference_of_endpoint_errors_v1"
    )
    assert dict(receipt.checks)[
        "old_identity_objective_not_equivalent"
    ] is True
    assert dict(receipt.checks)[
        "identity_sorr_separable_omco_not_equivalent"
    ] is True


def test_cmif_parameter_and_forward_contract_remain_exact(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    assert receipt.parameter_names == (
        "joint_hidden_bias",
        "joint_state_weight",
        "scalar_energy_weight",
    )
    assert receipt.formal_config_payload["forward_parameters"] == [
        "feature",
        "occupancy",
    ]
    assert receipt.formal_config_payload["output_shape"] == [
        1,
        1,
        8,
        12,
    ]
    assert receipt.formal_config_payload["state_unchanged"] is True
    assert receipt.formal_config_payload["auxiliary_outputs"] == 0


def test_pmope_reaches_all_cmif_parameters_without_an_update(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    probe = receipt.cmif_parameter_gradient_probe
    assert probe["probe_kind"] == (
        "generated_clean_pair_autograd_only"
    )
    assert probe["known_initial_multiplicative_latency_avoided"] is True
    assert probe["scalar_energy_weight_fixed_nonzero"] is True
    assert probe["all_parameter_gradients_finite"] is True
    assert probe["all_parameter_gradients_nonzero"] is True
    assert set(probe["gradient_contract"]) == {
        "joint_state_weight",
        "joint_hidden_bias",
        "scalar_energy_weight",
    }
    assert all(
        value["finite"] and value["nonzero"]
        for value in probe["gradient_contract"].values()
    )
    assert probe["model_state_unchanged"] is True
    assert probe["optimizer_constructed"] is False
    assert probe["optimizer_steps"] == 0


def test_receipt_has_no_data_or_training_authority(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    payload = receipt.canonical_payload()
    assert payload["runtime_splits"] == []
    assert payload["D_R_accessed"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["dataset_training_performed"] is False
    assert payload["synthetic_gradient_probe_optimizer_steps"] == 0
    assert payload["bounded_400_authorized"] is False
    assert payload["formal_800_authorized"] is False
    assert payload["full_CURE_authorized"] is False
    assert payload["cross_backbone_authorized"] is False


def test_implementation_binding_is_complete_and_current(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    assert tuple(
        path for path, _ in receipt.implementation_binding
    ) == COVERAGE_STATE_PMOPE_IMPLEMENTATION_PATHS
    assert all(
        len(digest) == 64
        for _, digest in receipt.implementation_binding
    )


def test_receipt_rejects_nested_evidence_mutation() -> None:
    receipt = run_coverage_state_pmope_dataset_free_gate()
    receipt.clean_probe["plus_gradient_exact_zero"] = False
    with pytest.raises(RuntimeError, match="evidence changed"):
        _ = receipt.all_pass
    with pytest.raises(RuntimeError, match="evidence changed"):
        _ = receipt.receipt_fingerprint


def test_receipt_rejects_a_tampered_check(
    receipt: CoverageStatePMOPEDatasetFreeReceipt,
) -> None:
    tampered = list(receipt.checks)
    name, value = tampered[0]
    tampered[0] = (name, not value)
    with pytest.raises(RuntimeError, match="evidence changed"):
        CoverageStatePMOPEDatasetFreeReceipt(
            formal_config_payload=deepcopy(
                receipt.formal_config_payload
            ),
            parameter_names=receipt.parameter_names,
            parameter_contract=receipt.parameter_contract,
            implementation_binding=receipt.implementation_binding,
            clean_probe=deepcopy(receipt.clean_probe),
            component_null_probe=deepcopy(
                receipt.component_null_probe
            ),
            zero_semantics_probe=deepcopy(
                receipt.zero_semantics_probe
            ),
            non_equivalence_probe=deepcopy(
                receipt.non_equivalence_probe
            ),
            cmif_parameter_gradient_probe=deepcopy(
                receipt.cmif_parameter_gradient_probe
            ),
            generated_replay_fingerprint=(
                receipt.generated_replay_fingerprint
            ),
            checks=tuple(tampered),
            evidence_fingerprint=receipt.evidence_fingerprint,
        )


def test_gate_restores_the_callers_cpu_rng_state() -> None:
    torch.manual_seed(1818)
    before = torch.random.get_rng_state().clone()
    run_coverage_state_pmope_dataset_free_gate()
    assert torch.equal(before, torch.random.get_rng_state())
