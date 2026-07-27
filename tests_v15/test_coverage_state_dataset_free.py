from __future__ import annotations

from dataclasses import replace

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.coverage_state_dataset_free import (
    COVERAGE_STATE_DATASET_FREE_CASES,
    COVERAGE_STATE_DATASET_FREE_SEEDS,
    COVERAGE_STATE_DATASET_FREE_SIZES,
    COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES,
    CoverageStateDatasetFreeReceipt,
    recompute_coverage_state_dataset_free_checks,
    run_coverage_state_dataset_free_gate,
)
from cure_lite.train.coverage_state_fused_step import (
    COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES,
)


@pytest.fixture(scope="module")
def receipt() -> CoverageStateDatasetFreeReceipt:
    return run_coverage_state_dataset_free_gate()


def test_dataset_free_geometry_matrix_is_complete_and_finite(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    expected = {
        (name, size, seed)
        for name in COVERAGE_STATE_DATASET_FREE_CASES
        for size in COVERAGE_STATE_DATASET_FREE_SIZES
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
    }
    actual = {
        (value.case_name, value.size, value.seed)
        for value in receipt.case_results
    }
    assert actual == expected
    assert len(receipt.case_results) == 16 * 2 * 3
    assert all(
        value.target_fields_finite
        and value.model_fields_finite
        and value.target_zero_level_exact
        and value.hard_union_exact
        for value in receipt.case_results
    )


def test_dataset_free_cases_cover_the_frozen_geometry_semantics(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    expected_target_pixels = {
        "one_pixel_target": (1, 1),
        "three_pixel_compact_target": (3, 3),
        "two_disconnected_targets": (2, 2),
        "target_at_image_edge": (2, 2),
        "target_near_invalid_barrier": (1, 1),
        "empty_state": (0, 0),
        "single_false_negative_island": (4, 4),
        "multiple_false_islands": (8, 8),
        "component_null_deletion": (0, 0),
        "identity_null": (0, 0),
        "same_feature_cell_multiple_occupancy_phases": (0, 0),
        "full_grid_deletion_hidden_by_scalar_projection": (0, 0),
        "clean_pair_with_natural_miss_already_present": (1, 17),
        "clutter_feature_peak_without_target": (0, 0),
        "low_rms_feature": (1, 1),
        "high_dynamic_range_feature": (1, 1),
    }
    for value in receipt.case_results:
        assert (
            value.target_pixels_plus,
            value.target_pixels_minus,
        ) == expected_target_pixels[value.case_name]

    empty = tuple(
        value
        for value in receipt.case_results
        if value.case_name == "empty_state"
    )
    assert len(empty) == 2 * 3
    assert all(
        value.predicted_negative_pixels == 0
        and value.predicted_negative_components == 0
        for value in empty
    )

    component = tuple(
        value
        for value in receipt.case_results
        if value.pair_kind == "component_null"
    )
    assert len(component) == 3 * 2 * 3
    assert all(
        value.component_new_negative_pixels == 0
        and value.component_new_negative_components == 0
        for value in component
    )

    identity = tuple(
        value
        for value in receipt.case_results
        if value.pair_kind == "identity_null"
    )
    assert len(identity) == 2 * 3
    assert all(
        value.scalar_visible is False
        and value.phase_visible is False
        and value.identity_field_exact is True
        for value in identity
    )


def test_dataset_free_representation_visibility_and_rf_are_explicit(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    optimizer_pairs = tuple(
        value
        for value in receipt.case_results
        if value.optimizer_eligible_pair
    )
    assert len(optimizer_pairs) == 2 * 2 * 3
    assert all(
        value.scalar_visible is True and value.phase_visible is True
        for value in optimizer_pairs
    )

    scalar_hidden_diagnostics = tuple(
        value
        for value in receipt.case_results
        if value.expected_scalar_hidden
    )
    assert len(scalar_hidden_diagnostics) == 2 * 2 * 3
    assert all(
        not value.optimizer_eligible_pair
        and value.scalar_visible is False
        and value.phase_visible is True
        and value.phase_roundtrip_exact is True
        and value.hidden_component_field_exact is True
        and value.target_response_pixels == 0
        for value in scalar_hidden_diagnostics
    )

    pair_rows = tuple(
        value
        for value in receipt.case_results
        if value.state_type == "pair"
    )
    assert len(pair_rows) == 5 * 2 * 3
    assert all(
        value.phase_roundtrip_exact is True
        and value.target_response_outside_scalar_rf_pixels == 0
        and value.target_response_outside_phase_rf_pixels == 0
        for value in pair_rows
    )


def test_dataset_free_training_matrix_checks_only_short_learnability(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    expected = {
        (seed, objective.value)
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
        for objective in COVERAGE_STATE_COMPLETION_ROOTED_MATCHED_OBJECTIVES
    }
    actual = {
        (value.seed, value.objective)
        for value in receipt.training_results
    }
    assert actual == expected
    assert len(receipt.training_results) == 3 * 3
    for value in receipt.training_results:
        assert value.updates == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
        assert value.forward_calls == 3
        assert value.backward_calls == 3
        assert value.optimizer_steps == 3
        assert value.logical_state_evaluations == 36
        assert value.factual_miss_target_pixels == 8
        assert value.factual_no_miss_target_pixels == 0
        assert value.losses_finite
        assert value.parameters_changed
        assert value.diagnostic_fields_finite
        assert value.identity_field_exact
        assert value.hidden_component_field_exact
        assert value.component_new_negative_pixels == 0
        assert value.component_new_negative_components == 0
        assert value.empty_negative_pixels == 0
        assert value.empty_negative_components == 0
        assert value.hard_union_exact
        latency = dict(value.first_nonzero_gradient_update)
        assert latency["phase_projection.weight"] == 0
        assert latency["phase_projection.bias"] == 0
        assert latency["input_projection.weight"] <= 2
        assert latency["spatial_mixing.weight"] <= 2

    for seed in COVERAGE_STATE_DATASET_FREE_SEEDS:
        rows = tuple(
            value
            for value in receipt.training_results
            if value.seed == seed
        )
        assert len({value.initial_model_fingerprint for value in rows}) == 1
        assert len({value.selection_fingerprint for value in rows}) == 1


def test_dataset_free_receipt_has_narrow_evidence_scope_and_fingerprint(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    payload = receipt.canonical_payload()
    assert receipt.all_pass
    assert receipt.status == "DATASET_FREE_GATE_PASS"
    assert len(receipt.checks) == 21
    assert all(passed for _, passed in receipt.checks)
    assert payload["scope"]["fixture_source"] == "generated_in_memory_only"
    assert payload["scope"]["selected_representation"] == "scalar_max"
    assert payload["scope"]["training_role"] == (
        "computational_learnability_and_early_gradient_only"
    )
    assert payload["data_access"] == {
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    assert payload["performance_claim_supported"] is False
    assert payload["formal_training_authorized"] is False
    assert receipt.receipt_fingerprint == stable_fingerprint(payload)
    assert len(receipt.receipt_fingerprint) == 64


def test_dataset_free_completion_root_probe_is_complete_and_decisive(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    assert {
        (value.size, value.seed)
        for value in receipt.completion_root_probes
    } == {
        (size, seed)
        for size in COVERAGE_STATE_DATASET_FREE_SIZES
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
    }
    assert len(receipt.completion_root_probes) == 6
    for value in receipt.completion_root_probes:
        assert value.target_pixels == 1
        assert value.target_negative_pixels_before_update == 0
        assert (
            value.response_sign_correct_pixels
            == value.response_sign_pixels
        )
        assert value.response_sign_pixels > 0
        assert float.fromhex(value.response_error_max_hex) <= 1.0e-6
        assert (
            float.fromhex(value.legacy_minus_gradient_max_hex)
            <= 1.0e-6
        )
        assert (
            float.fromhex(
                value.rooted_minus_target_gradient_min_hex
            )
            >= 1.0e-4
        )
        assert float.fromhex(value.exact_rooted_loss_hex) == 0.0
        assert (
            float.fromhex(value.component_null_rooted_loss_hex)
            == 0.0
        )


def test_dataset_free_receipt_rejects_scope_and_matrix_drift(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    failed_checks = (
        (receipt.checks[0][0], False),
        *receipt.checks[1:],
    )
    with pytest.raises(ValueError, match="recomputed results"):
        replace(receipt, checks=failed_checks)

    with pytest.raises(ValueError, match="evidence scope"):
        replace(receipt, D_R_accessed=True)
    with pytest.raises(ValueError, match="evidence scope"):
        replace(receipt, performance_claim_supported=True)
    with pytest.raises(ValueError, match="matrix is incomplete"):
        replace(
            receipt,
            case_results=(
                receipt.case_results[0],
                *receipt.case_results[:-1],
            ),
        )
    with pytest.raises(ValueError, match="matrix is incomplete"):
        replace(
            receipt,
            training_results=(
                receipt.training_results[0],
                *receipt.training_results[:-1],
            ),
        )
    with pytest.raises(ValueError, match="matrix is incomplete"):
        replace(
            receipt,
            completion_root_probes=(
                receipt.completion_root_probes[0],
                *receipt.completion_root_probes[:-1],
            ),
        )


def test_dataset_free_receipt_recomputes_checks_from_case_evidence(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    changed_cases = (
        replace(
            receipt.case_results[0],
            target_zero_level_exact=False,
        ),
        *receipt.case_results[1:],
    )
    with pytest.raises(ValueError, match="recomputed results"):
        replace(receipt, case_results=changed_cases)

    recomputed = recompute_coverage_state_dataset_free_checks(
        changed_cases,
        receipt.training_results,
        receipt.completion_root_probes,
    )
    failed = replace(
        receipt,
        case_results=changed_cases,
        checks=recomputed,
    )
    assert not failed.all_pass
    assert failed.status == "DATASET_FREE_GATE_FAIL"
    assert dict(failed.checks)["target_zero_level_exact"] is False


def test_dataset_free_receipt_recomputes_checks_from_training_evidence(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    changed_training = (
        replace(receipt.training_results[0], losses_finite=False),
        *receipt.training_results[1:],
    )
    with pytest.raises(ValueError, match="recomputed results"):
        replace(receipt, training_results=changed_training)

    recomputed = recompute_coverage_state_dataset_free_checks(
        receipt.case_results,
        changed_training,
        receipt.completion_root_probes,
    )
    failed = replace(
        receipt,
        training_results=changed_training,
        checks=recomputed,
    )
    assert not failed.all_pass
    assert (
        dict(failed.checks)["three_objectives_computationally_learnable"]
        is False
    )


def test_dataset_free_receipt_recomputes_checks_from_root_probe_evidence(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    changed_probes = (
        replace(
            receipt.completion_root_probes[0],
            target_negative_pixels_before_update=1,
        ),
        *receipt.completion_root_probes[1:],
    )
    with pytest.raises(ValueError, match="recomputed results"):
        replace(receipt, completion_root_probes=changed_probes)

    recomputed = recompute_coverage_state_dataset_free_checks(
        receipt.case_results,
        receipt.training_results,
        changed_probes,
    )
    failed = replace(
        receipt,
        completion_root_probes=changed_probes,
        checks=recomputed,
    )
    assert not failed.all_pass
    assert (
        dict(failed.checks)[
            "completion_root_probe_response_correct_without_crossing"
        ]
        is False
    )


def test_dataset_free_receipt_verify_rejects_post_init_tampering(
    receipt: CoverageStateDatasetFreeReceipt,
) -> None:
    tampered = replace(receipt)
    object.__setattr__(
        tampered,
        "checks",
        (
            (tampered.checks[0][0], False),
            *tampered.checks[1:],
        ),
    )
    with pytest.raises(ValueError, match="recomputed results"):
        tampered.verify()
    with pytest.raises(ValueError, match="recomputed results"):
        _ = tampered.all_pass
