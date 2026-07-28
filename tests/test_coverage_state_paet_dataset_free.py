from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from cure_lite.experiment.coverage_state_paet_dataset_free import (
    COVERAGE_STATE_PAET_DATASET_FREE_CHECK_NAMES,
    COVERAGE_STATE_PAET_DATASET_FREE_EXECUTION_SEED,
    COVERAGE_STATE_PAET_DATASET_FREE_SCHEMA,
    COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_PAET_IMPLEMENTATION_PATHS,
    CoverageStatePAETDatasetFreeReceipt,
    recompute_coverage_state_paet_dataset_free_checks,
    run_coverage_state_paet_dataset_free_gate,
)


@pytest.fixture(scope="module")
def receipts():
    before = torch.random.get_rng_state().clone()
    first = run_coverage_state_paet_dataset_free_gate()
    middle = torch.random.get_rng_state().clone()
    second = run_coverage_state_paet_dataset_free_gate()
    after = torch.random.get_rng_state().clone()
    return first, second, before, middle, after


def test_paet_dataset_free_gate_passes_exactly_fifteen_checks(
    receipts,
) -> None:
    first, second, before, middle, after = receipts

    assert first.all_pass
    assert second.all_pass
    assert tuple(name for name, _ in first.checks) == (
        COVERAGE_STATE_PAET_DATASET_FREE_CHECK_NAMES
    )
    assert len(first.checks) == 15
    assert all(value for _, value in first.checks)
    assert first.canonical_payload()["schema_version"] == (
        COVERAGE_STATE_PAET_DATASET_FREE_SCHEMA
    )
    assert first.canonical_payload()["execution_seed"] == (
        COVERAGE_STATE_PAET_DATASET_FREE_EXECUTION_SEED
    )
    assert torch.equal(before, middle)
    assert torch.equal(middle, after)


def test_paet_dataset_free_gate_is_exactly_replayable(receipts) -> None:
    first, second, *_ = receipts

    assert first.generated_replay_fingerprint == (
        second.generated_replay_fingerprint
    )
    assert first.evidence_fingerprint == second.evidence_fingerprint
    assert first.receipt_fingerprint == second.receipt_fingerprint
    assert first.canonical_payload() == second.canonical_payload()
    first.verify_unchanged()
    second.verify_unchanged()


def test_paet_phase_geometry_and_analytic_transport_are_explicit(
    receipts,
) -> None:
    receipt = receipts[0]
    offsets = receipt.probes["offset_and_order"]
    geometry = receipt.probes["transport_geometry"]
    ramp = receipt.probes["analytic_ramp"]

    assert offsets["axis_offsets_hex"] == [
        (-0.375).hex(),
        (-0.125).hex(),
        (0.125).hex(),
        (0.375).hex(),
    ]
    assert offsets["phase_count"] == 16
    assert offsets["phase_offsets_row_major_exact"] is True
    assert offsets["phase_pack_row_major_exact"] is True
    assert geometry["direct_pack_exact"] is True
    assert geometry["pixelshuffle_unpack_exact"] is True
    assert geometry["constant_upsample_exact"] is True
    assert geometry["constant_phase_exact"] is True
    assert ramp["interior_cell_count"] == 12
    assert ramp["maximum_abs_interior_error_hex"] == (0.0).hex()
    assert ramp["all_16_phase_values_unique"] is True
    assert ramp["phase_evidence_nondegenerate"] is True


def test_paet_keeps_shared_bfa_field_and_parameter_contract(
    receipts,
) -> None:
    receipt = receipts[0]
    reference = receipt.probes["efficient_reference"]
    flip = receipt.probes["flip_antisymmetry"]
    anchors = receipt.probes["zero_feature_and_additive"]
    parameters = receipt.probes["parameter_contract"]
    interface = receipt.probes["single_field_interface"]

    assert reference["efficient_reference_allclose"] is True
    assert reference["standalone_upsample_exact"] is True
    assert reference["standalone_phase_pack_exact"] is True
    assert reference["model_phase_evidence_nondegenerate"] is True
    assert flip["flip_involution_exact"] is True
    assert flip["native_odd_projection_exact"] is True
    assert flip["selected_odd_allclose"] is True
    assert flip["field_sum_two_anchor_allclose"] is True
    assert flip["expected_field_sum_hex"] == (1.8).hex()
    assert anchors["zero_feature_field_exact_anchor"] is True
    assert anchors["occupancy_path_field_exact_anchor"] is True
    assert anchors["feature_path_field_exact_anchor"] is True
    assert parameters["parameter_keys_and_shapes_exact"] is True
    assert parameters["seed42_initial_state_exact"] is True
    assert parameters["parameter_tensor_count"] == 3
    assert parameters["parameter_count"] == (
        COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
    )
    assert parameters["paet_state_fingerprint"] == (
        parameters["bfa_state_fingerprint"]
    )
    assert interface["forward_only_feature_and_occupancy"] is True
    assert interface["unexpected_role_keyword_rejected"] is True
    assert interface["output_is_one_tensor"] is True
    assert interface["only_pixel_shuffle_child"] is True
    assert interface["completion_is_bool_single_field"] is True


def test_paet_first_and_second_order_paths_reach_all_parameters(
    receipts,
) -> None:
    gradients = receipts[0].probes["gradients"]

    assert gradients["parameter_names"] == [
        "joint_state_weight",
        "joint_hidden_bias",
        "scalar_energy_weight",
    ]
    assert gradients["first_reaches_all_three_parameters"] is True
    assert gradients["second_reaches_all_three_parameters"] is True
    assert all(
        value["finite"] and value["nonzero"]
        for value in gradients["first_order"]
    )
    assert all(
        value["finite"] and value["nonzero"]
        for value in gradients["second_order"]
    )
    assert gradients["model_state_preserved"] is True
    assert gradients["parameter_grad_buffers_unretained"] is True
    assert gradients["backward_called"] is False
    assert gradients["optimizer_constructed"] is False


def test_paet_gate_has_no_runtime_data_or_tunable_transport(
    receipts,
) -> None:
    receipt = receipts[0]
    boundary = receipt.probes["static_boundary"]
    frozen = receipt.probes["no_tunable_transport"]
    payload = receipt.canonical_payload()

    assert boundary["interpolate_call_count"] == 1
    assert boundary["interpolate_arguments_fixed"] is True
    assert boundary["forbidden_calls_present"] == []
    assert boundary["forbidden_runtime_metadata_present"] == []
    assert boundary["parsed_python_sources"] == 2
    assert boundary["forbidden_imports"] == []
    assert boundary["forbidden_source_calls"] == []
    assert boundary["runtime_splits"] == []
    assert boundary["dataset_constructed"] is False
    assert boundary["cache_constructed"] is False
    assert boundary["optimizer_constructed"] is False
    assert boundary["training_performed"] is False
    assert frozen["added_config_fields"] == ["transport_policy"]
    assert frozen["removed_config_fields"] == []
    assert frozen["numeric_added_config_fields"] == []
    assert frozen["learned_transport_parameters"] == []
    assert frozen["learned_offsets"] is False
    assert frozen["temperature"] is None
    assert frozen["transport_scale"] is None
    assert frozen["transport_bias"] is None
    assert payload["runtime_splits"] == []
    assert payload["D_R_accessed"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["optimizer_constructed"] is False
    assert payload["training_performed"] is False
    assert payload["bounded_400_authorized"] is False


def test_paet_receipt_binds_current_implementation(receipts) -> None:
    receipt = receipts[0]

    assert tuple(
        name for name, _ in receipt.implementation_binding
    ) == COVERAGE_STATE_PAET_IMPLEMENTATION_PATHS
    assert all(
        len(digest) == 64
        for _, digest in receipt.implementation_binding
    )


def test_paet_recompute_fails_the_specific_tampered_claim(
    receipts,
) -> None:
    receipt = receipts[0]
    probes = deepcopy(receipt.probes)
    probes["analytic_ramp"]["analytic_ramp_interior_exact"] = False
    checks = dict(
        recompute_coverage_state_paet_dataset_free_checks(
            probes=probes,
            implementation_binding=receipt.implementation_binding,
            generated_replay_fingerprint=(
                receipt.generated_replay_fingerprint
            ),
        )
    )

    assert checks["05_analytic_ramp_interior_exact"] is False
    assert sum(not value for value in checks.values()) == 1


def test_paet_receipt_rejects_mutated_evidence(receipts) -> None:
    receipt = receipts[0]
    probes = deepcopy(receipt.probes)
    probes["no_tunable_transport"]["learned_offsets"] = True

    with pytest.raises(RuntimeError, match="evidence changed"):
        CoverageStatePAETDatasetFreeReceipt(
            probes=probes,
            implementation_binding=receipt.implementation_binding,
            generated_replay_fingerprint=(
                receipt.generated_replay_fingerprint
            ),
            checks=receipt.checks,
            evidence_fingerprint=receipt.evidence_fingerprint,
        )
