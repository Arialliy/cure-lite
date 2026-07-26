from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "polarity_recoverable_subpixel_vacancy_evidence_factorization_v6"
)
_CLOSURE = _PROTOCOL / "toy_gate_closure_receipt.json"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_v6_toy_closure_fingerprint_and_all_file_bindings() -> None:
    closure = _load(_CLOSURE)
    unsigned = dict(closure)
    fingerprint = unsigned.pop("receipt_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)

    result_binding = closure["result_binding"]
    result_path = _ROOT / result_binding["repo_path"]
    assert file_sha256(result_path) == result_binding["file_sha256"]
    result = _load(result_path)
    result_unsigned = dict(result)
    result_fingerprint = result_unsigned.pop("result_fingerprint")
    assert result_fingerprint == stable_fingerprint(result_unsigned)
    assert result_fingerprint == result_binding["result_fingerprint"]

    for binding in closure["protocol_bindings"].values():
        assert file_sha256(_ROOT / binding["repo_path"]) == (
            binding["file_sha256"]
        )
    for group_name in (
        "implementation_bindings",
        "verification_test_bindings",
    ):
        for repo_path, expected_sha256 in closure[group_name].items():
            assert file_sha256(_ROOT / repo_path) == expected_sha256


def test_v6_toy_closure_authorizes_code_not_real_data_execution() -> None:
    closure = _load(_CLOSURE)
    assert closure["phase_status"] == "FROZEN_TOY_GATE_PASS"
    assert closure["decision"] == "PRSVEF_V6_TOY_GATE_PASS"
    gate = closure["gate_summary"]
    assert gate["toy_gate_pass"] is True
    assert gate["passed_case_count"] == 3
    assert gate["failed_case_count"] == 0
    assert gate["bounded_code_creation_authorized"] is True
    assert gate["real_D_R_bounded_authorized"] is False

    boundary = closure["boundary"]
    assert boundary["D_R_accessed_by_toy_gate"] is False
    assert boundary["D_V_accessed"] is False
    assert boundary["D_T_accessed"] is False
    assert boundary["bounded_code_creation_authorized"] is True
    assert boundary["real_D_R_bounded_authorized"] is False
    assert boundary["real_D_R_status"] == "NOT_RUN_TOY_PHASE"
    assert boundary["detection_performance_evaluated"] is False
    assert boundary["formal_800_authorized"] is False
    assert boundary["automatic_retry_performed"] is False

    # These are historical toy-phase facts recorded by the frozen receipt.
    # Later additive bounded-code or run artifacts must not invalidate them.
