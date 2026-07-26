from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
import cure_lite.experiment.paired_tiny_target_representability as tiny


def _singleton_case(origin: tuple[int, int]) -> tiny.TinyTargetCase:
    shape = tiny.enumerate_tiny_target_shapes()[0]
    row_signature = stable_fingerprint(
        tiny._axis_equivalence_signature(origin[0], [0])
    )
    column_signature = stable_fingerprint(
        tiny._axis_equivalence_signature(origin[1], [0])
    )
    problem_fingerprint = tiny._compact_problem_fingerprint(
        shape, row_signature, column_signature
    )
    unsigned = {
        "schema_version": tiny.TINY_TARGET_CASE_SCHEMA,
        "shape_id": shape.shape_id,
        "representative_origin": list(origin),
        "row_origins": [origin[0]],
        "column_origins": [origin[1]],
        "row_axis_signature_fingerprint": row_signature,
        "column_axis_signature_fingerprint": column_signature,
        "problem_fingerprint": problem_fingerprint,
    }
    return tiny.TinyTargetCase(
        case_id=stable_fingerprint(unsigned),
        shape=shape,
        representative_origin=origin,
        row_origins=(origin[0],),
        column_origins=(origin[1],),
        row_axis_signature_fingerprint=row_signature,
        column_axis_signature_fingerprint=column_signature,
        problem_fingerprint=problem_fingerprint,
    )


def _certificate(
    case_id: str,
    *,
    status: str,
) -> tiny.TinyTargetCaseCertificate:
    solved = status != "INCONCLUSIVE"
    structural = status == "STRUCTURAL_FAIL"
    reason = None if status == "PASS" else "test"
    false_additions = 7 if structural else (0 if solved else None)
    margin = 0.1 if solved else None
    raw_fa = 7 / 65536 if structural else (0.0 if solved else None)
    metric = 0.0 if solved else None
    target = True if solved else None
    recall = 1.0 if solved else None
    active = ("0x0.0p+0",) if solved else ()
    positives = tuple((0, index) for index in range(7)) if structural else ()
    violations = ("raw_background_fa",) if structural else ()
    dense = "1" * 64 if solved else None
    witness = "2" * 64 if solved else None
    unsigned = {
        "schema_version": tiny.TINY_TARGET_CERTIFICATE_SCHEMA,
        "case_id": case_id,
        "case_status": status,
        "reason": reason,
        "irreducible_false_addition_pixels": false_additions,
        "localized_certifying_margin": margin,
        "target_pixel_recall": recall,
        "target_matched": target,
        "retention": recall,
        "retention_semantics": "target_pixel_recall",
        "stage_a_anchor_retention_applicable": False,
        "pixel_fa": metric,
        "raw_background_fa": raw_fa,
        "fp_components_per_mp": metric,
        "active_value_hex": list(active),
        "positive_background_pixels": [list(value) for value in positives],
        "budget_violations": list(violations),
        "bound_normalization_max_abs": 0.0 if solved else None,
        "dense_problem_fingerprint": dense,
        "witness_fingerprint": witness,
        "stage_1_solver": {"fake": True} if solved else {},
        "stage_2_solver": {"fake": True} if solved else {},
    }
    return tiny.TinyTargetCaseCertificate(
        case_id=case_id,
        case_status=status,
        reason=reason,
        irreducible_false_addition_pixels=false_additions,
        localized_certifying_margin=margin,
        target_pixel_recall=recall,
        target_matched=target,
        retention=recall,
        pixel_fa=metric,
        raw_background_fa=raw_fa,
        fp_components_per_mp=metric,
        active_value_hex=active,
        positive_background_pixels=positives,
        budget_violations=violations,
        bound_normalization_max_abs=0.0 if solved else None,
        dense_problem_fingerprint=dense,
        witness_fingerprint=witness,
        stage_1_solver={"fake": True} if solved else {},
        stage_2_solver={"fake": True} if solved else {},
        certificate_fingerprint=stable_fingerprint(unsigned),
    )


def _resign(
    certificate: tiny.TinyTargetCaseCertificate,
    **changes: object,
) -> tiny.TinyTargetCaseCertificate:
    unsigned = certificate.payload(unsigned=True)
    for key, value in changes.items():
        if key == "active_value_hex":
            unsigned[key] = list(value)
        elif key == "positive_background_pixels":
            unsigned[key] = [list(pixel) for pixel in value]
        elif key in {"budget_violations"}:
            unsigned[key] = list(value)
        else:
            unsigned[key] = value
    return replace(
        certificate,
        **changes,
        certificate_fingerprint=stable_fingerprint(unsigned),
    )


def test_shape_catalog_is_the_complete_oriented_cc8_area_one_to_three_set() -> None:
    shapes = tiny.enumerate_tiny_target_shapes()
    assert len(shapes) == 25
    assert Counter(shape.area for shape in shapes) == {1: 1, 2: 4, 3: 20}
    assert len({shape.shape_id for shape in shapes}) == 25
    assert all(min(row for row, _ in shape.pixels) == 0 for shape in shapes)
    assert all(min(column for _, column in shape.pixels) == 0 for shape in shapes)


def test_exact_axis_operator_preserves_constants_and_matches_pytorch() -> None:
    low = torch.arange(64, dtype=torch.float64).reshape(1, 1, 1, 64)
    actual = torch.nn.functional.interpolate(
        low,
        size=(1, 256),
        mode="bilinear",
        align_corners=False,
    )[0, 0, 0]
    expected = []
    for output_index in range(256):
        weights = tiny.axis_bilinear_weights(output_index)
        assert sum(value for _, value in weights) == 8
        expected.append(
            sum(
                float(low[0, 0, 0, source]) * numerator / 8.0
                for source, numerator in weights
            )
        )
    assert torch.equal(actual, torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize("offsets", ([0], [0, 1], [0, 2], [0, 1, 2]))
def test_axis_equivalence_is_exact_and_covers_each_origin_once(
    offsets: list[int],
) -> None:
    classes = tiny.build_axis_equivalence_classes(offsets)
    extent = max(offsets) + 1
    origins = [origin for item in classes for origin in item.origins]
    assert sorted(origins) == list(range(256 - extent + 1))
    assert len(origins) == len(set(origins))
    assert len({item.signature_fingerprint for item in classes}) == len(classes)
    assert all(
        stable_fingerprint(tiny._axis_equivalence_signature(origin, offsets))
        == item.signature_fingerprint
        for item in classes
        for origin in item.origins
    )


def test_rational_2d_reconstruction_matches_frozen_bilinear_operator() -> None:
    shape = next(
        shape
        for shape in tiny.enumerate_tiny_target_shapes()
        if shape.pixels == ((0, 0), (1, 1), (2, 2))
    )
    for origin in ((0, 0), (2, 2), (33, 34), (253, 253)):
        problem = tiny.build_representability_problem(shape, origin)
        values = np.linspace(
            -0.8, 0.9, num=len(problem.active_nodes), dtype=np.float64
        )
        rational = tiny.reconstruct_output_logits(problem, values)
        pytorch = tiny.torch_bilinear_output_from_active_values(
            problem, values
        ).numpy()
        # The rational coefficient tables are exact; the two floating-point
        # evaluation orders may differ by one float64 rounding unit.
        assert np.max(np.abs(rational - pytorch)) <= np.finfo(np.float64).eps


def test_full_case_catalog_covers_all_concrete_placements_without_duplication() -> None:
    catalog = tiny.build_tiny_target_case_catalog()
    assert len(catalog.shapes) == 25
    derived_count = sum(
        len(
            tiny.build_axis_equivalence_classes(
                [row for row, _ in shape.pixels]
            )
        )
        * len(
            tiny.build_axis_equivalence_classes(
                [column for _, column in shape.pixels]
            )
        )
        for shape in catalog.shapes
    )
    assert len(catalog.cases) == derived_count
    assert catalog.concrete_placement_count == 1_622_566
    assert sum(case.multiplicity for case in catalog.cases) == 1_622_566
    assert len({case.case_id for case in catalog.cases}) == len(catalog.cases)


def test_milp_records_certified_irreducible_boundary_and_interior_errors() -> None:
    pytest.importorskip("scipy.optimize")
    corner = tiny.solve_tiny_target_case(_singleton_case((0, 0)))
    boundary_phase = tiny.solve_tiny_target_case(_singleton_case((2, 2)))
    interior = tiny.solve_tiny_target_case(_singleton_case((34, 34)))

    assert corner.case_status == "PASS"
    assert corner.irreducible_false_addition_pixels == 3
    assert corner.raw_background_fa == 3 / 65536

    assert boundary_phase.case_status == "STRUCTURAL_FAIL"
    assert boundary_phase.irreducible_false_addition_pixels == 8
    assert boundary_phase.raw_background_fa == 8 / 65536
    assert boundary_phase.raw_background_fa > 1e-4

    assert interior.case_status == "PASS"
    assert interior.irreducible_false_addition_pixels == 0
    assert interior.raw_background_fa == 0.0
    tiny.verify_tiny_target_case_certificate(
        _singleton_case((0, 0)),
        corner,
    )
    tiny.verify_tiny_target_case_certificate(
        _singleton_case((2, 2)),
        boundary_phase,
    )
    tiny.verify_tiny_target_case_certificate(
        _singleton_case((34, 34)),
        interior,
    )


def test_all_case_decision_is_a_conjunction_with_inconclusive_precedence() -> None:
    catalog = tiny.build_tiny_target_case_catalog()
    passing = [_certificate(case.case_id, status="PASS") for case in catalog.cases]
    decision = tiny._aggregate_tiny_target_case_statuses(
        catalog,
        tuple(passing),
    )
    assert decision["status"] == "LATE_STATIC_GATE_PASS"
    assert decision["all_case_conjunction"] is True
    assert decision["training_authorized"] is False

    structural = list(passing)
    structural[0] = _certificate(
        structural[0].case_id, status="STRUCTURAL_FAIL"
    )
    decision = tiny._aggregate_tiny_target_case_statuses(
        catalog,
        tuple(structural),
    )
    assert decision["status"] == "STRUCTURAL_FAIL"
    assert decision["all_case_conjunction"] is False
    assert decision["failing_cases"][0]["multiplicity"] > 0
    assert decision["failing_concrete_placement_count"] > 0

    inconclusive = list(structural)
    inconclusive[1] = _certificate(
        inconclusive[1].case_id, status="INCONCLUSIVE"
    )
    decision = tiny._aggregate_tiny_target_case_statuses(
        catalog,
        tuple(inconclusive),
    )
    assert decision["status"] == "COMPUTATIONALLY_INCONCLUSIVE"
    assert decision["historical_wave_a_decision_may_change"] is False


def test_decision_rejects_self_hashed_but_unreplayable_pass_certificates() -> None:
    catalog = tiny.build_tiny_target_case_catalog()
    values = tuple(
        _certificate(case.case_id, status="PASS")
        for case in catalog.cases
    )
    with pytest.raises(ValueError, match="certificate|solver"):
        tiny.build_tiny_target_decision(catalog, values)


def test_exactness_zero_requires_exact_primal_dual_and_margin_zero() -> None:
    payload = {
        "objective": 0.0,
        "mip_dual_bound": 0.0,
        "reconstructed_objective": 0.0,
    }
    assert tiny._search_has_frozen_zero_certificate(
        margin=0.0,
        payload=payload,
    )
    assert not tiny._search_has_frozen_zero_certificate(
        margin=1e-12,
        payload=payload,
    )
    assert not tiny._search_has_frozen_zero_certificate(
        margin=0.0,
        payload={**payload, "mip_dual_bound": None},
    )


def test_strict_replay_rejects_resigned_witness_or_solver_trace_tampering() -> None:
    case = _singleton_case((0, 0))
    certificate = tiny.solve_tiny_target_case(case)
    changed_active = list(certificate.active_value_hex)
    changed_active[0] = float(-1.0).hex()
    witness_tampered = _resign(
        certificate,
        active_value_hex=tuple(changed_active),
    )
    with pytest.raises(ValueError, match="witness|threshold|metric"):
        tiny.verify_tiny_target_case_certificate(case, witness_tampered)

    changed_stage_2 = dict(certificate.stage_2_solver)
    changed_stage_2["selected_witness_source"] = "invented"
    trace_tampered = _resign(
        certificate,
        stage_2_solver=changed_stage_2,
    )
    with pytest.raises(ValueError, match="selected witness"):
        tiny.verify_tiny_target_case_certificate(case, trace_tampered)


def test_bound_solver_replay_rejects_complete_self_consistent_fake_optimum() -> None:
    case = _singleton_case((0, 0))
    certificate = tiny.solve_tiny_target_case(case)
    assert certificate.irreducible_false_addition_pixels == 3
    forged_stage_1 = dict(certificate.stage_1_solver)
    forged_stage_1.update(
        {
            "objective": 3.0,
            "mip_dual_bound": 3.0,
            "reconstructed_objective": 3.0,
            "reconstructed_background_count": 3,
            "reconstructed_optimality_error": 0.0,
            "objective_reconstruction_error_abs": 0.0,
        }
    )
    forged_stage_2 = dict(certificate.stage_2_solver)
    forged_stage_2["exactness_search_attempts"] = []
    forged = _resign(
        certificate,
        stage_1_solver=forged_stage_1,
        stage_2_solver=forged_stage_2,
    )

    # The serialized trace is internally consistent, but it is not an
    # independent optimality proof.
    tiny.verify_tiny_target_case_certificate(case, forged)
    with pytest.raises(ValueError, match="bound-solver replay"):
        tiny.replay_tiny_target_case_certificate(case, forged)


def test_solved_certificate_rejects_empty_solver_payloads() -> None:
    certificate = tiny.solve_tiny_target_case(_singleton_case((34, 34)))
    unsigned = certificate.payload(unsigned=True)
    unsigned["stage_1_solver"] = {}
    with pytest.raises(ValueError, match="solver payload"):
        replace(
            certificate,
            stage_1_solver={},
            certificate_fingerprint=stable_fingerprint(unsigned),
        )


def test_case_identity_rejects_problem_or_policy_drift() -> None:
    case = _singleton_case((34, 34))
    with pytest.raises(ValueError, match="case_id"):
        replace(case, problem_fingerprint="0" * 64)


def test_certificate_status_is_fail_closed() -> None:
    case_id = "a" * 64
    with pytest.raises(ValueError, match="status"):
        _certificate(case_id, status="UNKNOWN")


def test_case_rejects_out_of_grid_equivalence_members() -> None:
    case = _singleton_case((34, 34))
    with pytest.raises(ValueError, match="outside"):
        replace(
            case,
            row_origins=(34, 256),
            case_id=stable_fingerprint(
                {
                    **{
                        key: value
                        for key, value in case.payload().items()
                        if key not in {"case_id", "multiplicity"}
                    },
                    "row_origins": [34, 256],
                }
            ),
        )


def test_reconstruction_normalizes_only_within_frozen_tolerance() -> None:
    problem = tiny.build_representability_problem(
        tiny.enumerate_tiny_target_shapes()[0],
        (34, 34),
    )
    values = np.zeros(len(problem.active_nodes), dtype=np.float64)
    values[0] = -1.0 - np.finfo(np.float64).eps
    output = tiny.reconstruct_output_logits(problem, values)
    assert np.isfinite(output).all()
    values[0] = -1.0 - 2 * tiny.VERIFY_TOLERANCE
    with pytest.raises(ValueError, match="tolerance"):
        tiny.reconstruct_output_logits(problem, values)
