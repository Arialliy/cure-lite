"""Descriptive decomposition of the 209-to-206 P0-B coverage transition.

This module deliberately has no protocol, gate, decision, training, or
evaluation side effects.  It accepts already extracted D_R target records and
replays the frozen group-distinct coverage statistic under six explicitly
separated representation-fitting regimes.
"""

from __future__ import annotations

from math import ceil
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from .p0_protocol import P0OverlapConfig
from .p0_support import (
    _FeatureProjector,
    _TargetRecord,
    _fit_feature_projector,
    _higher_quantile,
    _project_feature,
    _robust_scale,
    _robust_scale_fit,
)


COVERAGE_TRANSITION_SCHEMA = "cure-lite-coverage-transition-v1"
_EXPECTED_FACTUAL = 32
_EXPECTED_OLD_LEGAL = 209
_EXPECTED_NEW_LEGAL = 206


def _identity(record: _TargetRecord) -> tuple[str, int, int | None]:
    return tuple(record.identity)


def _identity_list(
    identity: tuple[str, int, int | None],
) -> list[str | int | None]:
    return [identity[0], identity[1], identity[2]]


def _record_order(
    records: Sequence[_TargetRecord],
) -> list[tuple[str, int, int | None]]:
    return [_identity(record) for record in records]


def _validate_population(
    old_factual: Sequence[_TargetRecord],
    old_legal: Sequence[_TargetRecord],
    new_factual: Sequence[_TargetRecord],
    new_legal: Sequence[_TargetRecord],
    overlap: P0OverlapConfig,
) -> tuple[_TargetRecord, ...]:
    if overlap.knn_k != 5:
        raise ValueError("coverage transition requires frozen group-distinct k=5")
    if (
        len(old_factual) != _EXPECTED_FACTUAL
        or len(new_factual) != _EXPECTED_FACTUAL
        or len(old_legal) != _EXPECTED_OLD_LEGAL
        or len(new_legal) != _EXPECTED_NEW_LEGAL
    ):
        raise ValueError(
            "coverage transition requires 32 factual, 209 old legal, "
            "and 206 new legal targets"
        )
    if any(record.role != "factual" for record in (*old_factual, *new_factual)):
        raise ValueError("factual sequences contain a non-factual record")
    if any(record.role != "legal" for record in (*old_legal, *new_legal)):
        raise ValueError("legal sequences contain a non-legal record")
    for name, records in (
        ("old factual", old_factual),
        ("new factual", new_factual),
        ("old legal", old_legal),
        ("new legal", new_legal),
    ):
        identities = _record_order(records)
        if len(set(identities)) != len(identities):
            raise ValueError(f"{name} identities are not unique")
    if _record_order(old_factual) != _record_order(new_factual):
        raise ValueError("old/new factual identity order differs")
    new_ids = set(_record_order(new_legal))
    remaining = tuple(record for record in old_legal if _identity(record) in new_ids)
    if _record_order(remaining) != _record_order(new_legal):
        raise ValueError(
            "new legal sequence is not the order-preserving old-population subset"
        )
    if len(old_legal) - len(remaining) != 3:
        raise ValueError("coverage transition requires exactly three exclusions")
    return remaining


def _tensor_fields_equal(first: _TargetRecord, second: _TargetRecord) -> bool:
    return bool(
        first.sample_id == second.sample_id
        and first.group_id == second.group_id
        and first.role == second.role
        and torch.equal(first.hand, second.hand)
        and torch.equal(first.joint_feature_raw, second.joint_feature_raw)
        and torch.equal(first.joint_occupancy_raw, second.joint_occupancy_raw)
    )


def _raw_equality(
    old_factual: Sequence[_TargetRecord],
    remaining_old_legal: Sequence[_TargetRecord],
    new_factual: Sequence[_TargetRecord],
    new_legal: Sequence[_TargetRecord],
) -> dict[str, object]:
    factual_rows = [
        {
            "identity": _identity_list(_identity(old)),
            "equal": _tensor_fields_equal(old, new),
        }
        for old, new in zip(old_factual, new_factual, strict=True)
    ]
    legal_rows = [
        {
            "identity": _identity_list(_identity(old)),
            "equal": _tensor_fields_equal(old, new),
        }
        for old, new in zip(remaining_old_legal, new_legal, strict=True)
    ]
    return {
        "comparison": (
            "identity-aligned exact torch.equal over hand, raw feature, "
            "raw occupancy, and metadata"
        ),
        "factual_equal": sum(bool(row["equal"]) for row in factual_rows),
        "factual_total": len(factual_rows),
        "retained_legal_equal": sum(bool(row["equal"]) for row in legal_rows),
        "retained_legal_total": len(legal_rows),
        "all_equal": all(bool(row["equal"]) for row in (*factual_rows, *legal_rows)),
        "factual_mismatches": [
            row["identity"] for row in factual_rows if not row["equal"]
        ],
        "retained_legal_mismatches": [
            row["identity"] for row in legal_rows if not row["equal"]
        ],
    }


def _projector_payload(projector: _FeatureProjector) -> dict[str, object]:
    payload: dict[str, object] = {
        "raw_median": [float(value) for value in projector.raw_median.tolist()],
        "raw_scale": [float(value) for value in projector.raw_scale.tolist()],
        "pca_mean": [float(value) for value in projector.pca_mean.tolist()],
        "basis": [
            [float(value) for value in row] for row in projector.basis.tolist()
        ],
        "singular_values": [
            float(value) for value in projector.singular_values.tolist()
        ],
    }
    payload["parameter_fingerprint"] = stable_fingerprint(payload)
    return payload


def _joint_values(
    factual: Sequence[_TargetRecord],
    legal: Sequence[_TargetRecord],
    projector: _FeatureProjector,
) -> tuple[Tensor, Tensor]:
    factual_feature = torch.stack(
        [record.joint_feature_raw for record in factual]
    )
    legal_feature = torch.stack([record.joint_feature_raw for record in legal])
    factual_occupancy = torch.stack(
        [record.joint_occupancy_raw for record in factual]
    )
    legal_occupancy = torch.stack(
        [record.joint_occupancy_raw for record in legal]
    )
    return (
        torch.cat(
            (
                _project_feature(factual_feature, projector),
                factual_occupancy,
            ),
            dim=1,
        ),
        torch.cat(
            (
                _project_feature(legal_feature, projector),
                legal_occupancy,
            ),
            dim=1,
        ),
    )


def _identity_sort_key(
    identity: tuple[str, int, int | None],
) -> tuple[str, int, int]:
    return (
        identity[0],
        identity[1],
        -1 if identity[2] is None else identity[2],
    )


def _group_top_k(
    query: Tensor,
    query_group: str,
    legal_values: Tensor,
    legal: Sequence[_TargetRecord],
    *,
    k: int,
) -> list[dict[str, object]]:
    distances = torch.linalg.vector_norm(legal_values - query, dim=1)
    by_group: dict[
        str, tuple[float, tuple[str, int, int | None]]
    ] = {}
    for distance, record in zip(distances.tolist(), legal, strict=True):
        if record.group_id == query_group:
            continue
        candidate = (float(distance), _identity(record))
        previous = by_group.get(record.group_id)
        if previous is None or (
            candidate[0],
            _identity_sort_key(candidate[1]),
        ) < (
            previous[0],
            _identity_sort_key(previous[1]),
        ):
            by_group[record.group_id] = candidate
    if len(by_group) < k:
        raise RuntimeError("fewer than k group-distinct legal references")
    ordered = sorted(
        (
            (distance, group_id, identity)
            for group_id, (distance, identity) in by_group.items()
        ),
        key=lambda row: (
            row[0],
            row[1],
            _identity_sort_key(row[2]),
        ),
    )[:k]
    return [
        {
            "rank": rank,
            "distance": distance,
            "group_id": group_id,
            "identity": _identity_list(identity),
        }
        for rank, (distance, group_id, identity) in enumerate(ordered, 1)
    ]


def _route(
    *,
    name: str,
    factual: Sequence[_TargetRecord],
    legal: Sequence[_TargetRecord],
    factual_values: Tensor,
    legal_values: Tensor,
    overlap: P0OverlapConfig,
    projector_fingerprint: str,
    median: Tensor,
    scale: Tensor,
    reference_radius: float | None,
    fit_policy: str,
) -> dict[str, object]:
    factual_scaled = _robust_scale(factual_values, median, scale)
    legal_scaled = _robust_scale(legal_values, median, scale)
    legal_reference_rows: list[float] = []
    for index, record in enumerate(legal):
        top = _group_top_k(
            legal_scaled[index],
            record.group_id,
            legal_scaled,
            legal,
            k=overlap.knn_k,
        )
        legal_reference_rows.append(float(top[-1]["distance"]))
    fitted_radius = _higher_quantile(
        legal_reference_rows,
        overlap.legal_reference_quantile,
    )
    radius = fitted_radius if reference_radius is None else float(reference_radius)
    rows: list[dict[str, object]] = []
    covered = 0
    for index, record in enumerate(factual):
        top = _group_top_k(
            factual_scaled[index],
            record.group_id,
            legal_scaled,
            legal,
            k=overlap.knn_k,
        )
        distance = float(top[-1]["distance"])
        inside = distance <= radius
        covered += int(inside)
        rows.append(
            {
                "identity": _identity_list(_identity(record)),
                "group_id": record.group_id,
                "kth_distance": distance,
                "distance_over_radius": distance / radius,
                "covered": inside,
                "group_distinct_top_k": top,
            }
        )
    scale_payload = {
        "median": [float(value) for value in median.tolist()],
        "scale": [float(value) for value in scale.tolist()],
    }
    scale_payload["parameter_fingerprint"] = stable_fingerprint(scale_payload)
    required = ceil(overlap.coverage_minimum * len(factual))
    return {
        "route": name,
        "fit_policy": fit_policy,
        "legal_targets": len(legal),
        "legal_groups": len({record.group_id for record in legal}),
        "projector_parameter_fingerprint": projector_fingerprint,
        "outer_scale": scale_payload,
        "reference_radius": radius,
        "counterfactual_fitted_reference_radius": fitted_radius,
        "reference_radius_was_refitted": reference_radius is None,
        "legal_reference_kth_distance_summary": {
            "minimum": min(legal_reference_rows),
            "median": _higher_quantile(legal_reference_rows, 0.5),
            "maximum": max(legal_reference_rows),
        },
        "covered_factual_targets": covered,
        "required_covered_factual_targets": required,
        "coverage": covered / len(factual),
        "descriptive_threshold_pass": covered >= required,
        "factual_targets": rows,
    }


def _fit_outer(values: Tensor) -> tuple[Tensor, Tensor]:
    median, scale, _, _ = _robust_scale_fit(values)
    return median, scale


def _covered_identities(route: Mapping[str, object]) -> set[tuple[str, int]]:
    rows = route["factual_targets"]
    assert isinstance(rows, list)
    return {
        (str(row["identity"][0]), int(row["identity"][1]))
        for row in rows
        if bool(row["covered"])
    }


def _identity_pairs(
    identities: set[tuple[str, int]],
) -> list[list[str | int]]:
    return [
        [identity[0], identity[1]]
        for identity in sorted(identities)
    ]


def _projector_change_summary(
    old: _FeatureProjector,
    new: _FeatureProjector,
) -> dict[str, object]:
    old_payload = _projector_payload(old)
    new_payload = _projector_payload(new)
    median_changed = int(torch.count_nonzero(old.raw_median != new.raw_median))
    scale_changed = int(torch.count_nonzero(old.raw_scale != new.raw_scale))
    return {
        "old": old_payload,
        "new": new_payload,
        "basis_fingerprint_changed": (
            old_payload["parameter_fingerprint"]
            != new_payload["parameter_fingerprint"]
        ),
        "raw_median_dimensions_changed_exactly": median_changed,
        "raw_scale_dimensions_changed_exactly": scale_changed,
        "raw_dimensions": int(old.raw_median.numel()),
        "interpretation": (
            "feature-projector refit includes its raw robust scale and PCA "
            "subspace; this summary does not assign semantic causality to an "
            "excluded target"
        ),
    }


def build_coverage_transition(
    old_factual: Sequence[_TargetRecord],
    old_legal: Sequence[_TargetRecord],
    new_factual: Sequence[_TargetRecord],
    new_legal: Sequence[_TargetRecord],
    overlap: P0OverlapConfig,
) -> dict[str, object]:
    """Return a deterministic, descriptive O/A/A+R/B/C/D decomposition."""

    old_factual = tuple(old_factual)
    old_legal = tuple(old_legal)
    new_factual = tuple(new_factual)
    new_legal = tuple(new_legal)
    remaining_old_legal = _validate_population(
        old_factual,
        old_legal,
        new_factual,
        new_legal,
        overlap,
    )
    new_legal_ids = set(_record_order(new_legal))
    excluded = tuple(
        record for record in old_legal if _identity(record) not in new_legal_ids
    )

    old_projector = _fit_feature_projector(
        torch.stack([record.joint_feature_raw for record in old_legal]),
        overlap.joint_feature_components,
    )
    old_projector_payload = _projector_payload(old_projector)
    old_factual_joint, old_legal_joint = _joint_values(
        old_factual,
        old_legal,
        old_projector,
    )
    remaining_indices = [
        index
        for index, record in enumerate(old_legal)
        if _identity(record) in new_legal_ids
    ]
    remaining_old_joint = old_legal_joint[remaining_indices]
    old_median, old_scale = _fit_outer(old_legal_joint)

    route_o = _route(
        name="O",
        factual=old_factual,
        legal=old_legal,
        factual_values=old_factual_joint,
        legal_values=old_legal_joint,
        overlap=overlap,
        projector_fingerprint=str(
            old_projector_payload["parameter_fingerprint"]
        ),
        median=old_median,
        scale=old_scale,
        reference_radius=None,
        fit_policy="old-209-projector-outer-scale-and-radius",
    )
    route_a = _route(
        name="A",
        factual=old_factual,
        legal=remaining_old_legal,
        factual_values=old_factual_joint,
        legal_values=remaining_old_joint,
        overlap=overlap,
        projector_fingerprint=str(
            old_projector_payload["parameter_fingerprint"]
        ),
        median=old_median,
        scale=old_scale,
        reference_radius=float(route_o["reference_radius"]),
        fit_policy="delete-three-with-old-projector-scale-and-radius-fixed",
    )
    route_a_radius = _route(
        name="A_plus_R",
        factual=old_factual,
        legal=remaining_old_legal,
        factual_values=old_factual_joint,
        legal_values=remaining_old_joint,
        overlap=overlap,
        projector_fingerprint=str(
            old_projector_payload["parameter_fingerprint"]
        ),
        median=old_median,
        scale=old_scale,
        reference_radius=None,
        fit_policy="delete-three-with-old-projector-scale-and-refit-radius",
    )
    remaining_median, remaining_scale = _fit_outer(remaining_old_joint)
    route_b = _route(
        name="B",
        factual=old_factual,
        legal=remaining_old_legal,
        factual_values=old_factual_joint,
        legal_values=remaining_old_joint,
        overlap=overlap,
        projector_fingerprint=str(
            old_projector_payload["parameter_fingerprint"]
        ),
        median=remaining_median,
        scale=remaining_scale,
        reference_radius=None,
        fit_policy="old-projector-with-refit-outer-scale-and-radius",
    )

    refit_projector = _fit_feature_projector(
        torch.stack(
            [record.joint_feature_raw for record in remaining_old_legal]
        ),
        overlap.joint_feature_components,
    )
    refit_payload = _projector_payload(refit_projector)
    refit_factual_joint, refit_legal_joint = _joint_values(
        old_factual,
        remaining_old_legal,
        refit_projector,
    )
    refit_median, refit_scale = _fit_outer(refit_legal_joint)
    cross_factual_joint, cross_old_legal_joint = _joint_values(
        old_factual,
        old_legal,
        refit_projector,
    )
    route_c_cross = _route(
        name="C_cross_209_fit206",
        factual=old_factual,
        legal=old_legal,
        factual_values=cross_factual_joint,
        legal_values=cross_old_legal_joint,
        overlap=overlap,
        projector_fingerprint=str(refit_payload["parameter_fingerprint"]),
        median=refit_median,
        scale=refit_scale,
        reference_radius=None,
        fit_policy=(
            "fit-projector-and-outer-scale-on-206-then-evaluate-209-"
            "and-refit-cell-radius"
        ),
    )
    route_c = _route(
        name="C",
        factual=old_factual,
        legal=remaining_old_legal,
        factual_values=refit_factual_joint,
        legal_values=refit_legal_joint,
        overlap=overlap,
        projector_fingerprint=str(refit_payload["parameter_fingerprint"]),
        median=refit_median,
        scale=refit_scale,
        reference_radius=None,
        fit_policy="refit-feature-projector-outer-scale-and-radius-on-old-minus-three",
    )

    new_projector = _fit_feature_projector(
        torch.stack([record.joint_feature_raw for record in new_legal]),
        overlap.joint_feature_components,
    )
    new_payload = _projector_payload(new_projector)
    new_factual_joint, new_legal_joint = _joint_values(
        new_factual,
        new_legal,
        new_projector,
    )
    new_median, new_scale = _fit_outer(new_legal_joint)
    route_d = _route(
        name="D",
        factual=new_factual,
        legal=new_legal,
        factual_values=new_factual_joint,
        legal_values=new_legal_joint,
        overlap=overlap,
        projector_fingerprint=str(new_payload["parameter_fingerprint"]),
        median=new_median,
        scale=new_scale,
        reference_radius=None,
        fit_policy="full-new-geometry-safe-population",
    )

    old_rows = {
        (str(row["identity"][0]), int(row["identity"][1])): row
        for row in route_o["factual_targets"]
    }
    route_rows = {
        name: {
            (str(row["identity"][0]), int(row["identity"][1])): row
            for row in route["factual_targets"]
        }
        for name, route in (
            ("O", route_o),
            ("A", route_a),
            ("A_plus_R", route_a_radius),
            ("B", route_b),
            ("C_cross_209_fit206", route_c_cross),
            ("C", route_c),
            ("D", route_d),
        )
    }
    transitions: list[dict[str, object]] = []
    for identity in old_rows:
        old_covered = bool(route_rows["O"][identity]["covered"])
        new_covered = bool(route_rows["D"][identity]["covered"])
        status = (
            "retained_covered"
            if old_covered and new_covered
            else "lost_coverage"
            if old_covered
            else "gained_coverage"
            if new_covered
            else "retained_uncovered"
        )
        transitions.append(
            {
                "identity": [identity[0], identity[1], None],
                "transition": status,
                "routes": {
                    name: {
                        "covered": bool(rows[identity]["covered"]),
                        "kth_distance": float(rows[identity]["kth_distance"]),
                        "distance_over_radius": float(
                            rows[identity]["distance_over_radius"]
                        ),
                    }
                    for name, rows in route_rows.items()
                },
            }
        )

    direct_rows: list[dict[str, object]] = []
    old_route_rows = route_o["factual_targets"]
    for record in excluded:
        identity = _identity(record)
        affected: list[list[str | int | None]] = []
        for row in old_route_rows:
            top_ids = {
                tuple(item["identity"])
                for item in row["group_distinct_top_k"]
            }
            if identity in top_ids:
                affected.append(list(row["identity"]))
        direct_rows.append(
            {
                "excluded_identity": _identity_list(identity),
                "old_group_distinct_top5_query_count": len(affected),
                "old_group_distinct_top5_factual_identities": affected,
            }
        )

    base_covered = _covered_identities(route_o)
    individual_replays: list[dict[str, object]] = []
    exclusion_sets = [
        (f"single_{index + 1}", (record,))
        for index, record in enumerate(excluded)
    ]
    exclusion_sets.append(("all_three", excluded))
    for replay_name, replay_excluded in exclusion_sets:
        replay_ids = {_identity(record) for record in replay_excluded}
        replay_indices = [
            index
            for index, record in enumerate(old_legal)
            if _identity(record) not in replay_ids
        ]
        replay_legal = tuple(old_legal[index] for index in replay_indices)
        replay_joint = old_legal_joint[replay_indices]
        replay = _route(
            name=f"individual_exclusion_{replay_name}",
            factual=old_factual,
            legal=replay_legal,
            factual_values=old_factual_joint,
            legal_values=replay_joint,
            overlap=overlap,
            projector_fingerprint=str(
                old_projector_payload["parameter_fingerprint"]
            ),
            median=old_median,
            scale=old_scale,
            reference_radius=float(route_o["reference_radius"]),
            fit_policy=(
                "fixed-old-209-projector-outer-scale-and-radius-with-"
                "specified-legal-exclusion"
            ),
        )
        replay_covered = _covered_identities(replay)
        individual_replays.append(
            {
                "replay": replay_name,
                "excluded_identities": [
                    _identity_list(_identity(record))
                    for record in replay_excluded
                ],
                "legal_targets": len(replay_legal),
                "projector_parameter_fingerprint": replay[
                    "projector_parameter_fingerprint"
                ],
                "outer_scale_parameter_fingerprint": replay[
                    "outer_scale"
                ]["parameter_fingerprint"],
                "fixed_reference_radius": replay["reference_radius"],
                "covered_factual_targets": replay[
                    "covered_factual_targets"
                ],
                "covered_factual_identities": _identity_pairs(replay_covered),
                "lost_vs_O": _identity_pairs(base_covered - replay_covered),
                "gained_vs_O": _identity_pairs(replay_covered - base_covered),
            }
        )

    c_d_projectors_equal = (
        refit_payload["parameter_fingerprint"]
        == new_payload["parameter_fingerprint"]
    )
    c_d_arrays_equal = bool(
        torch.equal(refit_factual_joint, new_factual_joint)
        and torch.equal(refit_legal_joint, new_legal_joint)
    )
    result: dict[str, object] = {
        "schema_version": COVERAGE_TRANSITION_SCHEMA,
        "scope": {
            "split": "D_R",
            "descriptive_only": True,
            "changes_p0_gate": False,
            "authorizes_candidate_construction": False,
            "authorizes_training": False,
            "authorizes_d_v_access": False,
            "authorizes_full_cure": False,
        },
        "population": {
            "factual_targets": len(old_factual),
            "old_legal_targets": len(old_legal),
            "new_legal_targets": len(new_legal),
            "excluded_legal_targets": [
                _identity_list(_identity(record)) for record in excluded
            ],
        },
        "raw_state_equality": _raw_equality(
            old_factual,
            remaining_old_legal,
            new_factual,
            new_legal,
        ),
        "direct_top5_influence": direct_rows,
        "individual_exclusion_replay": {
            "fit_population": "old-209-legal",
            "projector": "fixed-from-O",
            "outer_scale": "fixed-from-O",
            "reference_radius": "fixed-from-O",
            "replays": individual_replays,
        },
        "routes": {
            "O": route_o,
            "A": route_a,
            "A_plus_R": route_a_radius,
            "B": route_b,
            "C_cross_209_fit206": route_c_cross,
            "C": route_c,
            "D": route_d,
        },
        "two_by_two_cell_mapping": {
            "factors": {
                "representation_fit_population": (
                    "feature-projector-and-outer-scale fit on legal 209 or 206"
                ),
                "evaluated_legal_population": (
                    "group-distinct reference/query support from legal 209 or 206"
                ),
                "cell_radius": "refit within each evaluated legal population",
            },
            "cells": {
                "legal209_fit209": "O",
                "legal206_fit209": "A_plus_R",
                "legal209_fit206": "C_cross_209_fit206",
                "legal206_fit206": "C",
            },
            "sequential_bridge_not_a_factorial_cell": "B",
        },
        "factual_transitions": transitions,
        "projector_summary": _projector_change_summary(
            old_projector,
            new_projector,
        ),
        "full_new_equivalence": {
            "c_and_d_projector_parameters_equal": c_d_projectors_equal,
            "c_and_d_joint_arrays_equal": c_d_arrays_equal,
            "c_and_d_covered_identities_equal": (
                _covered_identities(route_c) == _covered_identities(route_d)
            ),
        },
        "descriptive_attribution": {
            "direct_deletion_changed_covered_identities": (
                _covered_identities(route_o) != _covered_identities(route_a)
            ),
            "radius_only_changed_covered_identities": (
                _covered_identities(route_a)
                != _covered_identities(route_a_radius)
            ),
            "outer_scale_refit_changed_covered_identities": (
                _covered_identities(route_a_radius)
                != _covered_identities(route_b)
            ),
            "feature_projector_refit_changed_covered_identities": (
                _covered_identities(route_b) != _covered_identities(route_c)
            ),
            "full_new_changed_beyond_refit_only": (
                _covered_identities(route_c) != _covered_identities(route_d)
                or not c_d_projectors_equal
                or not c_d_arrays_equal
            ),
        },
        "interpretation_limits": [
            "This is a deterministic D_R-only descriptive replay, not a new gate.",
            "Feature-projector refit jointly changes its raw robust scale and PCA.",
            "Reference-radius values are not directly comparable across changed coordinates.",
            "No route authorizes candidate construction, training, D_V access, or Full CURE.",
        ],
    }
    result["receipt_fingerprint"] = stable_fingerprint(result)
    return result
