from __future__ import annotations

from dataclasses import replace
import json

import pytest

from cure_lite.coverage_state_sobolev import (
    CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
)
from cure_lite.experiment.coverage_state_dataset_free import (
    COVERAGE_STATE_DATASET_FREE_SEEDS,
    COVERAGE_STATE_DATASET_FREE_SIZES,
    COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES,
    COVERAGE_STATE_LEGACY_DATASET_FREE_FINGERPRINT,
    COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_SCHEMA,
    CoverageStateSupportOrientedDatasetFreeReceipt,
    run_coverage_state_support_oriented_dataset_free_gate,
)
from cure_lite.train.coverage_state_fused_step import (
    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
)


def _canonical_bytes(
    receipt: CoverageStateSupportOrientedDatasetFreeReceipt,
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
    CoverageStateSupportOrientedDatasetFreeReceipt,
    CoverageStateSupportOrientedDatasetFreeReceipt,
]:
    return (
        run_coverage_state_support_oriented_dataset_free_gate(),
        run_coverage_state_support_oriented_dataset_free_gate(),
    )


def test_support_oriented_gate_binds_unchanged_legacy_receipt(
    replay_pair: tuple[
        CoverageStateSupportOrientedDatasetFreeReceipt,
        CoverageStateSupportOrientedDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]

    assert receipt.legacy_receipt_fingerprint == (
        COVERAGE_STATE_LEGACY_DATASET_FREE_FINGERPRINT
    )
    assert receipt.legacy_receipt.receipt_fingerprint == (
        "44f85b45adc42eaefc79278d4c519aac40a5ed17034b6eb"
        "51576452ff4db935d"
    )
    assert receipt.legacy_receipt.all_pass
    assert receipt.canonical_payload()[
        "legacy_dataset_free_receipt_fingerprint"
    ] == COVERAGE_STATE_LEGACY_DATASET_FREE_FINGERPRINT


def test_support_oriented_probe_matrix_is_complete_and_decisive(
    replay_pair: tuple[
        CoverageStateSupportOrientedDatasetFreeReceipt,
        CoverageStateSupportOrientedDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]
    expected_keys = {
        (size, seed)
        for size in COVERAGE_STATE_DATASET_FREE_SIZES
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
    }

    assert {
        (value.size, value.seed)
        for value in receipt.support_oriented_probes
    } == expected_keys
    assert len(receipt.support_oriented_probes) == 6
    for value in receipt.support_oriented_probes:
        assert value.objective_policy == (
            CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
        )
        assert value.selector_pixels == value.expected_added_target_pixels
        assert value.selector_pixels > 0
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


def test_support_oriented_gate_runs_the_new_matched_suite(
    replay_pair: tuple[
        CoverageStateSupportOrientedDatasetFreeReceipt,
        CoverageStateSupportOrientedDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]
    expected_keys = {
        (seed, objective.value)
        for seed in COVERAGE_STATE_DATASET_FREE_SEEDS
        for objective in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
    }

    assert {
        (value.seed, value.objective)
        for value in receipt.training_results
    } == expected_keys
    assert len(receipt.training_results) == 9
    for value in receipt.training_results:
        assert value.updates == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
        assert value.forward_calls == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
        assert value.backward_calls == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
        assert value.optimizer_steps == COVERAGE_STATE_DATASET_FREE_TRAINING_UPDATES
        assert value.losses_finite
        assert value.parameters_changed


def test_support_oriented_receipt_is_narrow_and_all_checks_pass(
    replay_pair: tuple[
        CoverageStateSupportOrientedDatasetFreeReceipt,
        CoverageStateSupportOrientedDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]
    payload = receipt.canonical_payload()

    assert payload["schema_version"] == (
        COVERAGE_STATE_SUPPORT_ORIENTED_DATASET_FREE_SCHEMA
    )
    assert receipt.all_pass
    assert receipt.status == "SUPPORT_ORIENTED_DATASET_FREE_GATE_PASS"
    assert all(dict(receipt.checks).values())
    assert payload["scope"]["no_gate_hyperparameter_search"] is True
    assert payload["scope"]["selector_source"] == (
        "frozen_target_fields_strict_signs_only"
    )
    assert payload["data_access"] == {
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    assert payload["performance_claim_supported"] is False
    assert payload["formal_training_authorized"] is False


def test_support_oriented_gate_replays_with_identical_canonical_bytes(
    replay_pair: tuple[
        CoverageStateSupportOrientedDatasetFreeReceipt,
        CoverageStateSupportOrientedDatasetFreeReceipt,
    ],
) -> None:
    first, second = replay_pair

    assert _canonical_bytes(first) == _canonical_bytes(second)
    assert first.receipt_fingerprint == second.receipt_fingerprint


def test_support_oriented_receipt_rejects_probe_tampering(
    replay_pair: tuple[
        CoverageStateSupportOrientedDatasetFreeReceipt,
        CoverageStateSupportOrientedDatasetFreeReceipt,
    ],
) -> None:
    receipt = replay_pair[0]
    changed = (
        replace(
            receipt.support_oriented_probes[0],
            selector_exact=False,
        ),
        *receipt.support_oriented_probes[1:],
    )

    with pytest.raises(
        ValueError,
        match="support-oriented dataset-free checks changed",
    ):
        replace(receipt, support_oriented_probes=changed)
