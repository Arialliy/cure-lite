from __future__ import annotations

from dataclasses import replace
from math import cos, sin

import pytest
import torch

from cure_lite.experiment.coverage_transition import (
    COVERAGE_TRANSITION_SCHEMA,
    build_coverage_transition,
)
from cure_lite.experiment.p0_protocol import P0OverlapConfig
from cure_lite.experiment.p0_support import _TargetRecord


def _overlap() -> P0OverlapConfig:
    return P0OverlapConfig(
        factual_population="reachable-factual-misses",
        legal_population="decoder-visible-legal-targets",
        group_key="manifest.group_id",
        exclude_same_group_neighbors=True,
        handcrafted_descriptor_fields=("dummy_a", "dummy_b"),
        probability_clip=1e-6,
        ring_inner_radius=4,
        ring_outer_radius=12,
        joint_feature_components=6,
        joint_feature_residual=(
            "legal-subspace-reconstruction-l2-per-sqrt-dimension-v1"
        ),
        joint_occupancy_representation=(
            "raw-local-patch-plus-global-fraction-v1"
        ),
        joint_occupancy_patch_radius=2,
        knn_k=5,
        legal_reference_quantile=0.95,
        coverage_minimum=0.9,
        robust_scale_rule="median-mad-maxdev-constant-floor-v1",
        quantile_rule="sorted-higher-v1",
    )


def _feature(index: int, *, offset: float) -> torch.Tensor:
    value = (index + 1.0 + offset) / 211.0
    return torch.tensor(
        [
            value,
            value**2,
            value**3,
            sin(0.11 * index + offset),
            cos(0.07 * index - offset),
            float(index % 7) + 0.1 * offset,
            float((index * 3) % 11) - 0.2 * offset,
            float((index * 17) % 23) + 0.3 * offset,
        ],
        dtype=torch.float64,
    )


def _record(index: int, *, role: str) -> _TargetRecord:
    factual = role == "factual"
    prefix = "F" if factual else "L"
    offset = 0.35 if factual else 0.0
    return _TargetRecord(
        identity=(
            f"{prefix}{index:03d}",
            1,
            None if factual else 1,
        ),
        sample_id=f"{prefix}{index:03d}",
        group_id=f"group-{prefix}-{index:03d}",
        role=role,
        hand=torch.tensor(
            [float(index), float(index % 13)],
            dtype=torch.float64,
        ),
        joint_feature_raw=_feature(index, offset=offset),
        joint_occupancy_raw=torch.tensor(
            [float(index % 2), float(index % 5) / 5.0],
            dtype=torch.float64,
        ),
    )


@pytest.fixture(scope="module")
def populations() -> tuple[
    tuple[_TargetRecord, ...],
    tuple[_TargetRecord, ...],
    tuple[_TargetRecord, ...],
    tuple[_TargetRecord, ...],
]:
    old_factual = tuple(_record(index, role="factual") for index in range(32))
    old_legal = tuple(_record(index, role="legal") for index in range(209))
    excluded = {10, 80, 160}
    new_legal = tuple(
        record
        for index, record in enumerate(old_legal)
        if index not in excluded
    )
    return old_factual, old_legal, old_factual, new_legal


@pytest.fixture(scope="module")
def receipt(
    populations: tuple[
        tuple[_TargetRecord, ...],
        tuple[_TargetRecord, ...],
        tuple[_TargetRecord, ...],
        tuple[_TargetRecord, ...],
    ],
) -> dict[str, object]:
    return build_coverage_transition(*populations, _overlap())


def test_transition_has_frozen_descriptive_scope_and_routes(
    receipt: dict[str, object],
) -> None:
    assert receipt["schema_version"] == COVERAGE_TRANSITION_SCHEMA
    assert receipt["population"] == {
        "factual_targets": 32,
        "old_legal_targets": 209,
        "new_legal_targets": 206,
        "excluded_legal_targets": [
            ["L010", 1, 1],
            ["L080", 1, 1],
            ["L160", 1, 1],
        ],
    }
    scope = receipt["scope"]
    assert scope["split"] == "D_R"
    assert scope["descriptive_only"] is True
    assert all(
        scope[field] is False
        for field in (
            "changes_p0_gate",
            "authorizes_candidate_construction",
            "authorizes_training",
            "authorizes_d_v_access",
            "authorizes_full_cure",
        )
    )
    routes = receipt["routes"]
    assert tuple(routes) == (
        "O",
        "A",
        "A_plus_R",
        "B",
        "C_cross_209_fit206",
        "C",
        "D",
    )
    assert routes["O"]["legal_targets"] == 209
    assert routes["C_cross_209_fit206"]["legal_targets"] == 209
    assert all(
        routes[name]["legal_targets"] == 206
        for name in ("A", "A_plus_R", "B", "C", "D")
    )


def test_transition_replays_frozen_and_refitted_parameters(
    receipt: dict[str, object],
) -> None:
    routes = receipt["routes"]
    old_projector = routes["O"]["projector_parameter_fingerprint"]
    assert routes["A"]["projector_parameter_fingerprint"] == old_projector
    assert routes["A_plus_R"]["projector_parameter_fingerprint"] == old_projector
    assert routes["B"]["projector_parameter_fingerprint"] == old_projector
    assert (
        routes["C"]["projector_parameter_fingerprint"]
        == routes["D"]["projector_parameter_fingerprint"]
    )
    assert (
        routes["C_cross_209_fit206"]["projector_parameter_fingerprint"]
        == routes["C"]["projector_parameter_fingerprint"]
    )
    assert (
        routes["C_cross_209_fit206"]["outer_scale"]["parameter_fingerprint"]
        == routes["C"]["outer_scale"]["parameter_fingerprint"]
    )
    assert routes["A"]["reference_radius_was_refitted"] is False
    assert routes["A"]["reference_radius"] == routes["O"]["reference_radius"]
    assert routes["A_plus_R"]["reference_radius_was_refitted"] is True
    assert routes["B"]["reference_radius_was_refitted"] is True
    assert (
        routes["A"]["outer_scale"]["parameter_fingerprint"]
        == routes["O"]["outer_scale"]["parameter_fingerprint"]
    )
    assert (
        routes["A_plus_R"]["outer_scale"]["parameter_fingerprint"]
        == routes["O"]["outer_scale"]["parameter_fingerprint"]
    )
    assert receipt["full_new_equivalence"] == {
        "c_and_d_projector_parameters_equal": True,
        "c_and_d_joint_arrays_equal": True,
        "c_and_d_covered_identities_equal": True,
    }
    assert receipt["two_by_two_cell_mapping"]["cells"] == {
        "legal209_fit209": "O",
        "legal206_fit209": "A_plus_R",
        "legal209_fit206": "C_cross_209_fit206",
        "legal206_fit206": "C",
    }


def test_transition_audits_all_targets_and_group_distinct_top5(
    receipt: dict[str, object],
) -> None:
    equality = receipt["raw_state_equality"]
    assert equality["all_equal"] is True
    assert equality["factual_equal"] == equality["factual_total"] == 32
    assert (
        equality["retained_legal_equal"]
        == equality["retained_legal_total"]
        == 206
    )
    assert len(receipt["factual_transitions"]) == 32
    assert len(receipt["direct_top5_influence"]) == 3
    replay = receipt["individual_exclusion_replay"]
    assert replay["projector"] == "fixed-from-O"
    assert replay["outer_scale"] == "fixed-from-O"
    assert replay["reference_radius"] == "fixed-from-O"
    assert len(replay["replays"]) == 4
    old_route = receipt["routes"]["O"]
    old_projector = old_route["projector_parameter_fingerprint"]
    old_scale = old_route["outer_scale"]["parameter_fingerprint"]
    old_radius = old_route["reference_radius"]
    for row in replay["replays"]:
        assert row["projector_parameter_fingerprint"] == old_projector
        assert row["outer_scale_parameter_fingerprint"] == old_scale
        assert row["fixed_reference_radius"] == old_radius
        assert len(row["covered_factual_identities"]) == row[
            "covered_factual_targets"
        ]
    for route in receipt["routes"].values():
        assert len(route["factual_targets"]) == 32
        for row in route["factual_targets"]:
            neighbors = row["group_distinct_top_k"]
            assert len(neighbors) == 5
            assert [item["rank"] for item in neighbors] == [1, 2, 3, 4, 5]
            assert len({item["group_id"] for item in neighbors}) == 5
            assert row["kth_distance"] == neighbors[-1]["distance"]


def test_transition_is_deterministic_and_detects_raw_mismatch(
    populations: tuple[
        tuple[_TargetRecord, ...],
        tuple[_TargetRecord, ...],
        tuple[_TargetRecord, ...],
        tuple[_TargetRecord, ...],
    ],
    receipt: dict[str, object],
) -> None:
    repeated = build_coverage_transition(*populations, _overlap())
    assert repeated == receipt
    assert repeated["receipt_fingerprint"] == receipt["receipt_fingerprint"]

    old_factual, old_legal, new_factual, new_legal = populations
    altered = list(new_legal)
    altered[0] = replace(
        altered[0],
        joint_feature_raw=altered[0].joint_feature_raw
        + torch.tensor(
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        ),
    )
    mismatched = build_coverage_transition(
        old_factual,
        old_legal,
        new_factual,
        tuple(altered),
        _overlap(),
    )
    assert mismatched["raw_state_equality"]["all_equal"] is False
    assert mismatched["raw_state_equality"]["retained_legal_mismatches"] == [
        ["L000", 1, 1]
    ]
    assert mismatched["receipt_fingerprint"] != receipt["receipt_fingerprint"]


def test_transition_rejects_non_frozen_population_counts(
    populations: tuple[
        tuple[_TargetRecord, ...],
        tuple[_TargetRecord, ...],
        tuple[_TargetRecord, ...],
        tuple[_TargetRecord, ...],
    ],
) -> None:
    old_factual, old_legal, new_factual, new_legal = populations
    with pytest.raises(ValueError, match="32 factual, 209 old legal"):
        build_coverage_transition(
            old_factual[:-1],
            old_legal,
            new_factual[:-1],
            new_legal,
            _overlap(),
        )
