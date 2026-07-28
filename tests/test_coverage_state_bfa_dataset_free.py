from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from cure_lite.cache.schema import file_sha256
from cure_lite.experiment.coverage_state_bfa_dataset_free import (
    COVERAGE_STATE_BFA_DATASET_FREE_CHECK_NAMES,
    COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED,
    COVERAGE_STATE_BFA_DATASET_FREE_SCHEMA,
    COVERAGE_STATE_BFA_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_BFA_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_BFA_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_BFA_FORMAL_WIDTH,
    COVERAGE_STATE_BFA_IMPLEMENTATION_PATHS,
    COVERAGE_STATE_BFA_MARGIN,
    CoverageStateBFADatasetFreeReceipt,
    recompute_coverage_state_bfa_dataset_free_checks,
    run_coverage_state_bfa_dataset_free_gate,
)


@pytest.fixture(scope="module")
def receipt() -> CoverageStateBFADatasetFreeReceipt:
    result = run_coverage_state_bfa_dataset_free_gate()
    assert result.all_pass
    return result


def test_bfa_dataset_free_contract_is_the_frozen_v20_contract() -> None:
    assert COVERAGE_STATE_BFA_DATASET_FREE_SCHEMA == (
        "cure-lite-bfa-cmif-v20-dataset-free-receipt-v1"
    )
    assert COVERAGE_STATE_BFA_DATASET_FREE_EXECUTION_SEED == 200020
    assert COVERAGE_STATE_BFA_FORMAL_FEATURE_CHANNELS == 64
    assert COVERAGE_STATE_BFA_FORMAL_FEATURE_STRIDE == 4
    assert COVERAGE_STATE_BFA_FORMAL_WIDTH == 32
    assert COVERAGE_STATE_BFA_FORMAL_PARAMETER_COUNT == 64064
    assert COVERAGE_STATE_BFA_MARGIN == 0.225
    assert len(COVERAGE_STATE_BFA_DATASET_FREE_CHECK_NAMES) == 15
    assert len(set(COVERAGE_STATE_BFA_DATASET_FREE_CHECK_NAMES)) == 15
    assert COVERAGE_STATE_BFA_IMPLEMENTATION_PATHS[-1] == (
        "cure_lite/experiment/coverage_state_bfa_dataset_free.py"
    )


def test_bfa_dataset_free_gate_passes_exactly_fifteen_checks(
    receipt: CoverageStateBFADatasetFreeReceipt,
) -> None:
    assert tuple(name for name, _ in receipt.checks) == (
        COVERAGE_STATE_BFA_DATASET_FREE_CHECK_NAMES
    )
    assert len(receipt.checks) == 15
    assert all(value for _, value in receipt.checks)
    assert receipt.all_pass
    receipt.verify_unchanged()


def test_bfa_dataset_free_probes_cover_every_frozen_requirement(
    receipt: CoverageStateBFADatasetFreeReceipt,
) -> None:
    probes = receipt.probes
    assert probes["flip_involution"]["involution_exact"]
    assert probes["flip_involution"]["changed_coordinates"] == [
        [2, 2, 2]
    ]
    assert probes["reference_equivalence"][
        "all_elements_within_frozen_tolerance"
    ]
    assert probes["local_antisymmetry"]["interaction_antisymmetric"]
    assert probes["local_antisymmetry"][
        "field_sum_equals_two_anchor"
    ]
    assert probes["zero_feature"]["field_exact_anchor"]
    assert probes["pure_paths"]["both_pure_paths_silent"]
    assert probes["affine_equivalence"][
        "old_midpoint_equals_new_binary_flip_exact"
    ]
    assert probes["nonlinear_witness"]["curvature_nonzero"]
    assert probes["nonlinear_witness"]["zero_level_output_differs"]
    assert probes["interval_feasibility"][
        "all_three_simultaneously_feasible"
    ]
    assert probes["pmope_gradient"][
        "all_three_directions_correct_and_finite"
    ]
    assert probes["phase_roundtrip"]["roundtrip_exact"]
    assert probes["parameter_contract"]["parameter_count"] == 64064
    assert probes["parameter_contract"][
        "initial_state_byte_equal_to_cmif"
    ]
    assert probes["staged_gradient"]["first_scalar_gradient_nonzero"]
    assert probes["staged_gradient"]["all_second_gradients_nonzero"]
    assert probes["forward_interface"]["forward_parameters_exact"]
    assert probes["forward_interface"][
        "forbidden_metadata_parameters_absent"
    ]
    assert probes["static_boundary"]["forbidden_imports"] == []
    assert probes["static_boundary"]["forbidden_calls"] == []


def test_bfa_dataset_free_receipt_is_exactly_replayable_and_rng_neutral(
    receipt: CoverageStateBFADatasetFreeReceipt,
) -> None:
    before_rng = torch.random.get_rng_state().clone()
    replay = run_coverage_state_bfa_dataset_free_gate()
    after_rng = torch.random.get_rng_state()
    assert torch.equal(before_rng, after_rng)
    assert replay.probes == receipt.probes
    assert replay.checks == receipt.checks
    assert (
        replay.generated_replay_fingerprint
        == receipt.generated_replay_fingerprint
    )
    assert replay.evidence_fingerprint == receipt.evidence_fingerprint
    assert replay.canonical_payload() == receipt.canonical_payload()
    assert replay.receipt_fingerprint == receipt.receipt_fingerprint


def test_bfa_dataset_free_canonical_payload_closes_runtime_boundaries(
    receipt: CoverageStateBFADatasetFreeReceipt,
) -> None:
    payload = receipt.canonical_payload()
    assert payload["schema_version"] == (
        COVERAGE_STATE_BFA_DATASET_FREE_SCHEMA
    )
    assert payload["model"] == "BFA-CMIF"
    assert payload["version"] == "v20"
    assert payload["input_interface"] == ["F_b", "O"]
    assert payload["check_count"] == 15
    assert payload["checks"] == dict(receipt.checks)
    assert payload["all_pass"] is True
    assert payload["runtime_splits"] == []
    assert payload["D_R_accessed"] is False
    assert payload["D_V_accessed"] is False
    assert payload["D_T_accessed"] is False
    assert payload["cache_artifact_accessed"] is False
    assert payload["model_artifact_accessed"] is False
    assert payload["optimizer_constructed"] is False
    assert payload["optimizer_steps"] == 0
    assert payload["parameter_updates"] == 0
    assert payload["training_performed"] is False
    assert payload["D_R_gate_authorized"] is True
    assert payload["bounded_400_authorized"] is False
    assert payload["formal_800_authorized"] is False


def test_bfa_dataset_free_implementation_binding_hashes_current_sources(
    receipt: CoverageStateBFADatasetFreeReceipt,
) -> None:
    root = Path(__file__).resolve().parents[1]
    assert tuple(
        path for path, _ in receipt.implementation_binding
    ) == COVERAGE_STATE_BFA_IMPLEMENTATION_PATHS
    for relative, digest in receipt.implementation_binding:
        assert digest == file_sha256(root / relative)
        assert len(digest) == 64


def test_bfa_dataset_free_receipt_detects_nested_evidence_mutation(
    receipt: CoverageStateBFADatasetFreeReceipt,
) -> None:
    changed = deepcopy(receipt)
    changed.probes["flip_involution"]["involution_exact"] = False
    with pytest.raises(
        RuntimeError,
        match="evidence changed after creation",
    ):
        changed.verify_unchanged()
    with pytest.raises(RuntimeError):
        _ = changed.all_pass
    with pytest.raises(RuntimeError):
        changed.canonical_payload()


def test_bfa_dataset_free_recompute_fails_tampered_evidence(
    receipt: CoverageStateBFADatasetFreeReceipt,
) -> None:
    probes = deepcopy(receipt.probes)
    probes["nonlinear_witness"]["curvature_nonzero"] = False
    checks = dict(
        recompute_coverage_state_bfa_dataset_free_checks(
            probes=probes,
            implementation_binding=receipt.implementation_binding,
            generated_replay_fingerprint=(
                receipt.generated_replay_fingerprint
            ),
        )
    )
    assert checks[
        "08_nonlinear_difference_zero_level_witness"
    ] is False
    assert checks["15_no_runtime_data_or_optimizer_path"] is False

    incomplete = deepcopy(probes)
    del incomplete["static_boundary"]
    incomplete_checks = (
        recompute_coverage_state_bfa_dataset_free_checks(
            probes=incomplete,
            implementation_binding=receipt.implementation_binding,
            generated_replay_fingerprint=(
                receipt.generated_replay_fingerprint
            ),
        )
    )
    assert len(incomplete_checks) == 15
    assert not any(value for _, value in incomplete_checks)
