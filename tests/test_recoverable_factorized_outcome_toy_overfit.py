from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import stable_fingerprint
from tools.evaluate_recoverable_factorized_toy_gate import evaluate


_ROOT = Path(__file__).resolve().parents[1]
_RESULT = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "polarity_recoverable_subpixel_vacancy_evidence_factorization_v6"
    / "toy_gate_result.json"
)


def test_recoverable_operator_exactly_replays_the_frozen_toy_pass() -> None:
    frozen = json.loads(_RESULT.read_text(encoding="utf-8"))
    observed = evaluate()

    assert observed == frozen
    unsigned = dict(frozen)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)

    assert frozen["decision"] == "PRSVEF_V6_TOY_GATE_PASS"
    assert frozen["all_pass"] is True
    assert frozen["passed_case_count"] == 3
    assert frozen["failed_case_count"] == 0
    assert frozen["passed_cases"] == [
        "one_pixel",
        "two_pixels",
        "three_pixels",
    ]
    assert frozen["failed_cases"] == []

    for row in frozen["cases"]:
        assert row["passed"] is True
        assert row["failed_checks"] == []
        assert all(row["checks"].values())
        assert all(row["endpoint_gradient_contract"].values())

    assert frozen["implementation_contract_pass"] is True
    assert frozen["bounded_code_creation_authorized"] is True
    assert frozen["real_D_R_bounded_authorized"] is False
    assert frozen["real_D_R_status"] == "NOT_RUN_TOY_PHASE"
    assert frozen["D_R_accessed"] is False
    assert frozen["D_V_accessed"] is False
    assert frozen["D_T_accessed"] is False
    assert frozen["detection_performance_evaluated"] is False
    assert frozen["formal_800_authorized"] is False
