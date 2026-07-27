from __future__ import annotations

from dataclasses import replace
import json

import pytest

from cure_lite.coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
)
from cure_lite.coverage_state_sobolev import (
    CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
)
from cure_lite.experiment.coverage_state_dataset_free import (
    COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES,
    COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT,
    COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SCHEMA,
    COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS,
    COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SIZES,
    COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_FINGERPRINT,
    CoverageStatePhasePreservingDatasetFreeReceipt,
    run_coverage_state_phase_preserving_dataset_free_gate,
)
from cure_lite.train.coverage_state_fused_step import (
    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
)


def _canonical_bytes(
    receipt: CoverageStatePhasePreservingDatasetFreeReceipt,
) -> bytes:
    return json.dumps(
        receipt.canonical_payload(),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


@pytest.fixture(scope="module")
def replay_pair() -> tuple[
    CoverageStatePhasePreservingDatasetFreeReceipt,
    CoverageStatePhasePreservingDatasetFreeReceipt,
]:
    return (
        run_coverage_state_phase_preserving_dataset_free_gate(),
        run_coverage_state_phase_preserving_dataset_free_gate(),
    )


def test_ppce_gate_binds_unchanged_support_oriented_receipt(
    replay_pair: tuple[
        CoverageStatePhasePreservingDatasetFreeReceipt,
        CoverageStatePhasePreservingDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]

    assert receipt.support_oriented_receipt_fingerprint == (
        COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_FINGERPRINT
    )
    assert receipt.support_oriented_receipt.receipt_fingerprint == (
        "56f56912359c5b12e10110323f01aeced279c1934f04080e5fe473f82c4d7c35"
    )
    assert receipt.support_oriented_receipt.all_pass
    assert receipt.canonical_payload()[
        "support_oriented_dataset_free_receipt_fingerprint"
    ] == COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_FINGERPRINT


def test_ppce_architecture_probe_closes_the_phase_contract(
    replay_pair: tuple[
        CoverageStatePhasePreservingDatasetFreeReceipt,
        CoverageStatePhasePreservingDatasetFreeReceipt,
    ],
) -> None:
    value = replay_pair[0].architecture_probe

    assert value.stride == 4
    assert value.phase_channel_count == 16
    assert value.roundtrip_case_count == 5
    assert value.roundtrip_exact
    assert value.phase_index_checks == 16
    assert value.phase_index_exact
    assert value.diagonal_alignment_checks == 16
    assert value.diagonal_alignment_exact
    assert value.scalar_projection_collision_exact
    assert value.phase_encoding_separates_collision
    assert value.ppce_state_separates_collision
    assert value.module_names == (
        "input_projection",
        "spatial_mixing",
        "phase_projection",
        "pixel_shuffle",
    )
    assert value.single_path_exact
    assert value.legacy_parameter_count == 19536
    assert value.ppce_parameter_count == 23856
    assert value.expected_ppce_parameter_count == 23856
    assert value.parameter_formula_exact
    assert value.parameter_delta == 4320
    assert value.expected_parameter_delta == 4320
    assert value.initial_positive_field_exact
    assert value.initial_completion_empty
    assert value.phase_occupancy_bool
    assert value.phase_occupancy_contiguous
    assert value.coverage_policy == (
        CSLF_PHASE_PRESERVING_COVERAGE_POLICY
    )


def test_ppce_preserves_frozen_target_sdf_and_measure_geometry(
    replay_pair: tuple[
        CoverageStatePhasePreservingDatasetFreeReceipt,
        CoverageStatePhasePreservingDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]

    assert {value.seed for value in receipt.geometry_probes} == set(
        COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
    )
    assert len(receipt.geometry_probes) == 3
    for value in receipt.geometry_probes:
        assert value.target_fields_exact
        assert value.sdf_fields_exact
        assert value.integration_measures_exact
        assert value.geometry_exact
        assert value.geometry_unchanged_after_ppce_forward
        assert value.support_oriented_inverse_exact
        assert value.target_field_fingerprint == (
            value.recomputed_target_field_fingerprint
        )
        assert value.target_field_fingerprint == (
            value.post_forward_target_field_fingerprint
        )
        assert value.sdf_fingerprint == value.recomputed_sdf_fingerprint
        assert value.sdf_fingerprint == value.post_forward_sdf_fingerprint
        assert value.integration_measure_fingerprint == (
            value.recomputed_integration_measure_fingerprint
        )
        assert value.integration_measure_fingerprint == (
            value.post_forward_integration_measure_fingerprint
        )
        assert value.geometry_fingerprint == (
            value.recomputed_geometry_fingerprint
        )
        assert value.geometry_fingerprint == (
            value.post_forward_geometry_fingerprint
        )


def test_ppce_gate_rechecks_sorr_selector_null_and_fixed_point(
    replay_pair: tuple[
        CoverageStatePhasePreservingDatasetFreeReceipt,
        CoverageStatePhasePreservingDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]
    expected = {
        (size, seed)
        for size in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SIZES
        for seed in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
    }

    assert {
        (value.size, value.seed)
        for value in receipt.support_oriented_probes
    } == expected
    assert len(receipt.support_oriented_probes) == 3
    for value in receipt.support_oriented_probes:
        assert value.objective_policy == (
            CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
        )
        assert value.selector_pixels == value.expected_added_target_pixels
        assert value.selector_exact
        assert value.root_inside_exact
        assert value.root_outside_exact
        assert value.response_exact
        assert value.direct_minus_gradient_nonzero
        assert value.identity_null_selector_empty
        assert value.identity_null_exact
        assert value.component_null_selector_empty
        assert value.component_null_exact
        assert value.fixed_point_zero
        assert value.fixed_point_gradients_zero
        assert value.gradients_finite
        assert value.boundary_gradients_finite


def test_ppce_gate_runs_three_matched_short_training_objectives(
    replay_pair: tuple[
        CoverageStatePhasePreservingDatasetFreeReceipt,
        CoverageStatePhasePreservingDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]
    expected = {
        (seed, objective.value)
        for seed in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS
        for objective in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
    }
    expected_parameter_names = {
        "input_projection.weight",
        "spatial_mixing.weight",
        "phase_projection.weight",
        "phase_projection.bias",
    }

    assert {
        (value.seed, value.objective)
        for value in receipt.training_results
    } == expected
    assert len(receipt.training_results) == 9
    for seed in COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SEEDS:
        rows = tuple(
            value
            for value in receipt.training_results
            if value.seed == seed
        )
        assert len(rows) == 3
        assert len(
            {value.initial_model_fingerprint for value in rows}
        ) == 1
        assert len({value.selection_fingerprint for value in rows}) == 1
    for value in receipt.training_results:
        assert value.updates == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
        assert value.forward_calls == 3
        assert value.backward_calls == 3
        assert value.optimizer_steps == 3
        assert value.logical_state_evaluations == 36
        assert value.losses_finite
        assert value.parameters_changed
        assert value.diagnostic_fields_finite
        assert value.identity_field_exact
        assert not value.hidden_component_field_exact
        assert value.empty_negative_pixels == 0
        assert value.empty_negative_components == 0
        assert value.hard_union_exact
        assert set(dict(value.first_nonzero_gradient_update)) == (
            expected_parameter_names
        )
        assert all(
            0 <= update < COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
            for _, update in value.first_nonzero_gradient_update
        )


def test_ppce_receipt_is_narrow_and_all_checks_pass(
    replay_pair: tuple[
        CoverageStatePhasePreservingDatasetFreeReceipt,
        CoverageStatePhasePreservingDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]
    payload = receipt.canonical_payload()

    assert payload["schema_version"] == (
        COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_SCHEMA
    )
    assert receipt.all_pass
    assert receipt.status == "PHASE_PRESERVING_DATASET_FREE_GATE_PASS"
    assert all(dict(receipt.checks).values())
    assert payload["scope"]["fixture_source"] == "generated_in_memory_only"
    assert payload["scope"]["changed_model_coordinate"] == (
        "lossless_phase_preserving_occupancy_encoding_only"
    )
    assert payload["scope"]["target_sdf_measure_policy"] == (
        "unchanged_and_fingerprinted"
    )
    assert payload["scope"]["no_gate_hyperparameter_search"] is True
    assert payload["data_access"] == {
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    assert payload["performance_claim_supported"] is False
    assert payload["formal_training_authorized"] is False


def test_ppce_gate_replays_with_identical_canonical_bytes_and_fingerprint(
    replay_pair: tuple[
        CoverageStatePhasePreservingDatasetFreeReceipt,
        CoverageStatePhasePreservingDatasetFreeReceipt,
    ],
) -> None:
    first, second = replay_pair

    assert _canonical_bytes(first) == _canonical_bytes(second)
    assert first.receipt_fingerprint == second.receipt_fingerprint
    assert first.receipt_fingerprint == (
        COVERAGE_STATE_PHASE_PRESERVING_DATASET_FREE_FINGERPRINT
    )
    assert first.receipt_fingerprint == (
        "3d0bf1c771966f04f96319bb3605d1aff90827843153ca8e89ba8965c5a79d2b"
    )


def test_ppce_receipt_rejects_architecture_and_geometry_tampering(
    replay_pair: tuple[
        CoverageStatePhasePreservingDatasetFreeReceipt,
        CoverageStatePhasePreservingDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]

    with pytest.raises(
        ValueError,
        match="phase-preserving dataset-free checks changed",
    ):
        replace(
            receipt,
            architecture_probe=replace(
                receipt.architecture_probe,
                diagonal_alignment_exact=False,
            ),
        )
    changed_geometry = (
        replace(
            receipt.geometry_probes[0],
            support_oriented_inverse_exact=False,
        ),
    )
    with pytest.raises(
        ValueError,
        match="phase-preserving dataset-free checks changed",
    ):
        replace(receipt, geometry_probes=changed_geometry)
