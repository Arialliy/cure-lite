from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "directed_subpixel_vacancy_evidence_factorization_v5"
)
_CLOSURE = _PROTOCOL / "toy_gate_closure_receipt.json"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_v5_toy_negative_closure_fingerprint_and_bindings() -> None:
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

    for binding_name in (
        "design_document",
        "proposal",
        "bounded_config",
        "negative_result_report",
    ):
        binding = closure["protocol_bindings"][binding_name]
        assert file_sha256(_ROOT / binding["repo_path"]) == (
            binding["file_sha256"]
        )

    for group_name in (
        "v5_runtime_files",
        "toy_fixture",
        "verification_tests",
    ):
        for repo_path, expected_sha256 in closure[
            "implementation_bindings"
        ][group_name].items():
            assert file_sha256(_ROOT / repo_path) == expected_sha256


def test_v5_toy_negative_closure_preserves_the_stop_rule() -> None:
    closure = _load(_CLOSURE)
    assert closure["phase_status"] == "FROZEN_VALID_TOY_GATE_NEGATIVE"
    assert closure["decision"] == "DSVEF_V5_TOY_GATE_FAIL"
    assert closure["gate_summary"]["toy_gate_pass"] is False
    assert closure["gate_summary"]["passed_case_count"] == 0
    assert closure["gate_summary"]["failed_case_count"] == 3
    assert closure["gate_summary"]["overall_code_gate_pass"] is False

    boundary = closure["boundary"]
    assert boundary["D_R_accessed_by_toy_gate"] is False
    assert boundary["D_V_accessed"] is False
    assert boundary["D_T_accessed"] is False
    assert boundary["real_D_R_status"] == "NOT_RUN_BY_TOY_STOP_RULE"
    assert boundary["real_D_R_authorized"] is False
    assert boundary["real_D_R_cli_created"] is False
    assert boundary["detection_performance_evaluated"] is False
    assert boundary["formal_800_authorized"] is False
    assert boundary["automatic_retry_performed"] is False

    assert not (
        _ROOT / "tools" / "run_directed_factorized_outcome_bounded.py"
    ).exists()
    assert not (_PROTOCOL / "COMPLETE.json").exists()
