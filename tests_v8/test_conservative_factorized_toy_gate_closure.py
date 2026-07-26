from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conserving_subpixel_evidence_allocation_v8"
)
CLOSURE = PROTOCOL / "toy_gate_closure_receipt.json"


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _verified_fingerprint(
    value: dict[str, object],
    field: str,
) -> str:
    unsigned = dict(value)
    observed = unsigned.pop(field)
    assert isinstance(observed, str)
    assert observed == stable_fingerprint(unsigned)
    return observed


def test_v8_toy_closure_is_complete_and_authorizes_only_dry_code() -> None:
    closure = _object(CLOSURE)
    closure_fingerprint = _verified_fingerprint(
        closure,
        "receipt_fingerprint",
    )

    assert closure["schema_version"] == (
        "cure-lite-cc-sea-v8-toy-gate-closure-v1"
    )
    assert closure["method_id"] == "cc_sea_v8"
    assert closure["decision"] == (
        "CC_SEA_V8_TOY_GATE_PASS_AND_DRY_RUN_CODE_AUTHORIZED"
    )
    assert len(closure_fingerprint) == 64

    for name in ("proposal", "toy_config", "toy_gate_result"):
        binding = closure["protocol_bindings"][name]
        path = ROOT / binding["repo_path"]
        assert file_sha256(path) == binding["file_sha256"]
        value = _object(path)
        fingerprint_field = binding["fingerprint_field"]
        assert (
            _verified_fingerprint(value, fingerprint_field)
            == binding["fingerprint"]
        )

    result = _object(PROTOCOL / "toy_gate_result.json")
    assert result["decision"] == "CC_SEA_V8_TOY_GATE_PASS"
    assert result["all_pass"] is True
    assert result["passed_case_count"] == 6
    assert result["failed_case_count"] == 0

    replay = closure["replay_evidence"]
    assert replay["independent_process_count"] == 3
    assert replay["all_byte_identical"] is True
    assert replay["all_file_sha256"] == [
        "da6b6b376abe8fb8707a17a56a12a67ed3a2397d346f2cdfd970e8ffab34556c"
    ]
    assert replay["result_fingerprint"] == (
        "2b0d14732b9b83c166e3e0c4934e230cae0d6e7f19b45c51089ba899c35de19a"
    )

    for repo_path, expected in closure["software_bindings"].items():
        assert file_sha256(ROOT / repo_path) == expected
    for repo_path, expected in closure["test_bindings"].items():
        assert file_sha256(ROOT / repo_path) == expected

    evidence = closure["test_evidence"]
    assert evidence["v8_preclosure"]["passed"] is True
    assert evidence["v8_preclosure"]["failed"] == 0
    assert evidence["v7_frozen_closure_regression"]["passed"] is True
    assert evidence["v7_frozen_closure_regression"]["failed"] == 0

    authorization = closure["authorization"]
    assert authorization == {
        "dry_run_code_creation": True,
        "dry_run_execution": False,
        "real_D_R_bounded_code_creation": False,
        "real_D_R_access": False,
        "D_V_access": False,
        "D_T_access": False,
        "formal_800": False,
        "full_CURE": False,
        "cross_detector": False,
    }
