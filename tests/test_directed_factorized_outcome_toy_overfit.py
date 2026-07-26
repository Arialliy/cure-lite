from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import stable_fingerprint
from tools.evaluate_directed_factorized_toy_gate import evaluate


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FROZEN_RESULT = (
    _REPO_ROOT
    / "protocols"
    / "IRSTD-1K"
    / "directed_subpixel_vacancy_evidence_factorization_v5"
    / "toy_gate_result.json"
)


def test_directed_activation_exactly_replays_the_frozen_toy_failure() -> None:
    """Keep the unchanged v5 gate failure visible and reproducible."""

    frozen = json.loads(_FROZEN_RESULT.read_text(encoding="utf-8"))
    observed = evaluate()

    assert observed == frozen
    unsigned = dict(frozen)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)

    assert frozen["decision"] == "DSVEF_V5_TOY_GATE_FAIL"
    assert frozen["all_pass"] is False
    assert frozen["passed_case_count"] == 0
    assert frozen["failed_case_count"] == 3
    assert frozen["passed_cases"] == []
    assert frozen["failed_cases"] == [
        "one_pixel",
        "two_pixels",
        "three_pixels",
    ]

    rows = {row["case_id"]: row for row in frozen["cases"]}
    assert rows["one_pixel"]["failed_checks"] == [
        "factual_miss_target",
        "plus_completion",
        "total_loss",
    ]
    assert rows["two_pixels"]["failed_checks"] == [
        "clean_D",
        "factual_miss_target",
        "plus_completion",
        "total_loss",
    ]
    assert rows["three_pixels"]["failed_checks"] == [
        "clean_D",
        "factual_miss_target",
        "plus_completion",
        "total_loss",
    ]

    assert frozen["implementation_contract_pass"] is True
    assert frozen["real_D_R_authorized"] is False
    assert frozen["real_D_R_status"] == "NOT_RUN_BY_TOY_STOP_RULE"
    assert frozen["D_R_accessed"] is False
    assert frozen["D_V_accessed"] is False
    assert frozen["D_T_accessed"] is False
    assert frozen["detection_performance_evaluated"] is False
    assert frozen["formal_800_authorized"] is False
