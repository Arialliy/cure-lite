from __future__ import annotations

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.paired_formal_decision import (
    FormalMethodEvidence,
)
from cure_lite.experiment.phase_resolved_real_formal_decision import (
    PFCR_FORMAL_COMPARATORS,
    assess_pfcr_formal_d_v_gate,
)


COMPARISON = stable_fingerprint({"kind": "comparison"})
PROTOCOL = stable_fingerprint({"kind": "pfcr-reveal"})


def _row(
    method: str,
    seed: int,
    *,
    true_targets: int,
    recovered: int,
    retention: float = 1.0,
    pixel_fa: float = 2.0e-5,
    raw_background_fa: float = 3.0e-5,
    components: float = 4.0,
    budget_violation: bool = False,
) -> FormalMethodEvidence:
    return FormalMethodEvidence(
        method=method,
        seed=seed,
        total_targets=170,
        true_targets=true_targets,
        pd=true_targets / 170,
        total_anchor_misses=23,
        recovered_anchor_misses=recovered,
        retention=retention,
        pixel_fa=pixel_fa,
        raw_background_fa=raw_background_fa,
        fp_components_per_mp=components,
        budget_violation=budget_violation,
        comparison_protocol_fingerprint=COMPARISON,
        result_fingerprint=stable_fingerprint(
            {
                "method": method,
                "seed": seed,
                "true_targets": true_targets,
                "recovered": recovered,
                "retention": retention,
                "pixel_fa": pixel_fa,
                "raw_background_fa": raw_background_fa,
                "components": components,
                "budget_violation": budget_violation,
            }
        ),
    )


def _evidence(
    *,
    seed42_pfcr: tuple[int, int] = (156, 9),
    seed43_pfcr: tuple[int, int] = (154, 7),
    seed43_retention: float = 1.0,
    seed43_budget_violation: bool = False,
) -> list[FormalMethodEvidence]:
    rows: list[FormalMethodEvidence] = []
    comparator_values = {
        42: {
            "Base@B": (150, 3),
            "F": (154, 7),
            "F×": (149, 2),
            "U": (151, 4),
            "paired_difference": (147, 0),
            "independent_endpoint": (154, 7),
        },
        43: {
            "Base@B": (150, 3),
            "F": (152, 5),
            "F×": (147, 0),
            "U": (152, 5),
            "paired_difference": (152, 5),
            "independent_endpoint": (152, 5),
        },
    }
    for seed, proposed in (
        (42, seed42_pfcr),
        (43, seed43_pfcr),
    ):
        rows.append(
            _row(
                "PFCR",
                seed,
                true_targets=proposed[0],
                recovered=proposed[1],
                retention=(
                    seed43_retention if seed == 43 else 1.0
                ),
                budget_violation=(
                    seed43_budget_violation if seed == 43 else False
                ),
            )
        )
        for method in PFCR_FORMAL_COMPARATORS:
            true_targets, recovered = comparator_values[seed][method]
            rows.append(
                _row(
                    method,
                    seed,
                    true_targets=true_targets,
                    recovered=recovered,
                )
            )
    return rows


def _assess(
    evidence: list[FormalMethodEvidence],
) -> dict[str, object]:
    return assess_pfcr_formal_d_v_gate(
        evidence,
        protocol_fingerprint=PROTOCOL,
        comparison_protocol_fingerprint=COMPARISON,
    )


def test_pfcr_gate_passes_exact_plus_two_boundary_for_both_seeds() -> None:
    decision = _assess(_evidence())

    assert decision["all_seeds_pass"] is True
    assert decision["status"] == "PFCR_D_V_GATE_PASS"
    assert decision["next_action"] == "FROZEN_CONFIRMATION_ONLY"
    assert decision["authorizes_frozen_confirmation"] is True
    assert decision["authorizes_full_cure"] is False
    assert decision["authorizes_cross_backbone"] is False
    assert decision["D_T_accessed"] is False
    assert [
        item["true_target_margin"]
        for item in decision["seed_decisions"]
    ] == [2, 2]
    assert [
        item["recovered_anchor_miss_margin"]
        for item in decision["seed_decisions"]
    ] == [2, 2]


def test_one_seed_plus_one_cannot_be_hidden_by_other_seed() -> None:
    decision = _assess(
        _evidence(
            seed42_pfcr=(160, 13),
            seed43_pfcr=(153, 6),
        )
    )

    assert decision["per_seed_not_mean"] is True
    assert decision["all_seeds_pass"] is False
    assert decision["status"] == "PFCR_D_V_GATE_FAIL"
    assert decision["next_action"] == "STOP_AND_PRESERVE_EVIDENCE"
    assert [
        item["pass"] for item in decision["seed_decisions"]
    ] == [True, False]
    assert decision["authorizes_frozen_confirmation"] is False


@pytest.mark.parametrize(
    ("retention", "budget_violation", "seed43_pfcr"),
    (
        (146 / 147, False, (153, 7)),
        (1.0, True, (154, 7)),
    ),
)
def test_pfcr_gate_rejects_retention_or_budget_failure(
    retention: float,
    budget_violation: bool,
    seed43_pfcr: tuple[int, int],
) -> None:
    decision = _assess(
        _evidence(
            seed43_retention=retention,
            seed43_budget_violation=budget_violation,
            seed43_pfcr=seed43_pfcr,
        )
    )

    assert decision["all_seeds_pass"] is False
    seed43 = decision["seed_decisions"][1]
    assert (
        seed43["checks"][
            "all_budget_and_retention_constraints_pass"
        ]
        is False
    )


def test_pfcr_gate_requires_exact_fourteen_row_evidence_set() -> None:
    rows = _evidence()
    with pytest.raises(ValueError, match="every frozen comparator"):
        _assess(rows[:-1])

    duplicate = [*rows, rows[-1]]
    with pytest.raises(ValueError, match="every frozen comparator"):
        _assess(duplicate)


def test_pfcr_gate_rejects_comparison_protocol_mismatch() -> None:
    rows = _evidence()
    rows[-1] = FormalMethodEvidence(
        **{
            **rows[-1].canonical_payload(),
            "comparison_protocol_fingerprint": stable_fingerprint(
                {"kind": "other"}
            ),
        }
    )

    with pytest.raises(ValueError, match="common comparison"):
        _assess(rows)


def test_pfcr_gate_rejects_forged_true_target_count_identity() -> None:
    rows = _evidence()
    original = rows[0]
    rows[0] = FormalMethodEvidence(
        **{
            **original.canonical_payload(),
            "true_targets": original.true_targets - 1,
            "pd": (original.true_targets - 1) / 170,
        }
    )

    with pytest.raises(ValueError, match="integer identity"):
        _assess(rows)
