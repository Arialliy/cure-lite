from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from tools import run_crossing_factorized_outcome_bounded as runner


_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "continuously_recoverable_log_vacancy_evidence_crossing_v7"
)
_CLOSURE = _PROTOCOL / "bounded_implementation_closure_receipt.json"

_EXPECTED_BINDINGS = {
    "bounded_implementation_proposal": {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
            "bounded_implementation_proposal_receipt.json"
        ),
        "file_sha256": (
            "65a45dc6d73d8cbf6bcb2c6b6204251f3583e0354ba3161b633f1547fbaa11dd"
        ),
        "fingerprint_field": "receipt_fingerprint",
        "fingerprint": (
            "d33f710348dec255fd73790b3c97c643472d115d3098e1469409f7dd57fad896"
        ),
    },
    "bounded_config": {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
            "bounded_config.json"
        ),
        "file_sha256": (
            "352c0c235134c1017b851854278255c2c678973929d3fda614389392502c4b96"
        ),
        "fingerprint_field": "config_fingerprint",
        "fingerprint": (
            "9bdc7f5567065c02d37cc82f94b5bc49c589dfee271487f4cbce7dd831c45818"
        ),
    },
    "bounded_dry_run_config": {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
            "bounded_dry_run_config.json"
        ),
        "file_sha256": (
            "709f72bc4d17798be4fecb01f96afb1b91a9fb39f6a5da80315a71b6b501e55c"
        ),
        "fingerprint_field": "config_fingerprint",
        "fingerprint": (
            "d5421a162822ad9962b9790a10c49c4bfe8cd7844c88c4ec5e80a7ca54559e97"
        ),
    },
    "bounded_dry_run_result": {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
            "bounded_dry_run_result.json"
        ),
        "file_sha256": (
            "01f98d35602942887e1f3003894be92beb428802b7989d4b2bbd2d04756ee490"
        ),
        "fingerprint_field": "result_fingerprint",
        "fingerprint": (
            "47cf682cf16023a8c14a468e2d7a83e0630a2a1bf02ee8d69c38254baee02993"
        ),
    },
    "toy_gate_closure": {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
            "toy_gate_closure_receipt.json"
        ),
        "file_sha256": (
            "25c3317045533f4116b8873d892fcd2c0e866d3e991843a4c0c8e872142f0fe5"
        ),
        "fingerprint_field": "receipt_fingerprint",
        "fingerprint": (
            "f95573edd8b842980d5b175b1aac8caf753f6c279342da8e29f54e165b1e255f"
        ),
    },
}

_DRY_RESULT_SHA256 = (
    "01f98d35602942887e1f3003894be92beb428802b7989d4b2bbd2d04756ee490"
)
_DRY_RESULT_FINGERPRINT = (
    "47cf682cf16023a8c14a468e2d7a83e0630a2a1bf02ee8d69c38254baee02993"
)
_DRY_RESULT_BYTE_COUNT = 7864
_SYNC_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
    "sync_benchmark_result_v2.json"
)
_SYNC_SHA256 = (
    "b73613881ce3bb530f450a62721ceb532e7aaa6881ace76945751d672d5744ad"
)
_SYNC_FINGERPRINT = (
    "0ce0df9ad5be90b4730aaea739c848d8564d462116540d3e9ce5e1cd7afd9742"
)
_WRAPPER_REPO_PATH = "tools/run_with_gpu_temperature_control.py"
_WRAPPER_SHA256 = (
    "026b751fbb59530721da1436af32f3bc924c9ed2ab3576df062a45bca7ec5e86"
)
_WRAPPER_TEST_REPO_PATH = "tests/test_gpu_temperature_control.py"
_REAL_RUNNER_TEST_REPO_PATH = (
    "tests/test_run_crossing_factorized_outcome_bounded_cli.py"
)
_SELF_REPO_PATH = (
    "tests/test_crossing_factorized_bounded_implementation_closure.py"
)

_IMPLEMENTATION_TEST_REPO_PATHS = (
    "tests/test_crossing_factorized_outcome_bounded.py",
    "tests/test_dry_run_crossing_factorized_outcome_bounded_cli.py",
    _REAL_RUNNER_TEST_REPO_PATH,
    _SELF_REPO_PATH,
    _WRAPPER_TEST_REPO_PATH,
)
_FOCUSED_TEST_REPO_PATHS = (
    "tests/test_crossing_factorized_config.py",
    "tests/test_crossing_factorized_decoder.py",
    "tests/test_crossing_factorized_model.py",
    "tests/test_crossing_factorized_outcome_bounded.py",
    "tests/test_crossing_factorized_outcome_toy_overfit.py",
    "tests/test_crossing_factorized_toy_gate_closure.py",
    "tests/test_crossing_factorized_sync_benchmark.py",
    "tests/test_crossing_factorized_bounded_protocol.py",
    "tests/test_dry_run_crossing_factorized_outcome_bounded_cli.py",
    _REAL_RUNNER_TEST_REPO_PATH,
    _WRAPPER_TEST_REPO_PATH,
)


def _load(path: Path) -> dict[str, object]:
    assert path.is_file()
    assert not path.is_symlink()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _unsigned(
    payload: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    unsigned = dict(payload)
    fingerprint = unsigned.pop(fingerprint_field)
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64
    assert fingerprint == stable_fingerprint(unsigned)
    return unsigned


def _repo_file_mapping(paths: tuple[str, ...]) -> dict[str, str]:
    return {
        path: file_sha256(_ROOT / path)
        for path in sorted(paths)
    }


def _all_test_file_mapping() -> dict[str, str]:
    return {
        path.relative_to(_ROOT).as_posix(): file_sha256(path)
        for path in sorted((_ROOT / "tests").glob("test_*.py"))
    }


def _assert_protocol_bindings(closure: Mapping[str, object]) -> None:
    protocol = closure["protocol_bindings"]
    assert isinstance(protocol, dict)
    assert set(protocol) == set(_EXPECTED_BINDINGS)
    for name, expected in _EXPECTED_BINDINGS.items():
        binding = protocol[name]
        assert isinstance(binding, dict)
        fingerprint_field = str(expected["fingerprint_field"])
        assert binding == {
            "repo_path": expected["repo_path"],
            "file_sha256": expected["file_sha256"],
            fingerprint_field: expected["fingerprint"],
        }
        path = _ROOT / str(expected["repo_path"])
        payload = _load(path)
        assert file_sha256(path) == expected["file_sha256"]
        assert payload[fingerprint_field] == expected["fingerprint"]
        _unsigned(payload, fingerprint_field=fingerprint_field)


def _assert_dry_replay_binding(closure: Mapping[str, object]) -> None:
    dry = closure["dry_run_result_binding"]
    assert isinstance(dry, dict)
    assert dry["repo_path"] == runner.DRY_RESULT_REPO_PATH
    assert dry["file_sha256"] == _DRY_RESULT_SHA256
    assert dry["canonical_sha256"] == _DRY_RESULT_SHA256
    assert dry["result_fingerprint"] == _DRY_RESULT_FINGERPRINT
    assert dry["canonical_byte_count"] == _DRY_RESULT_BYTE_COUNT
    assert dry["process_replay_count"] == 2
    assert dry["independent_process_file_sha256"] == [
        _DRY_RESULT_SHA256,
        _DRY_RESULT_SHA256,
    ]
    assert dry["independent_temporary_output_directories"] is True
    assert dry["byte_identical"] is True
    assert dry["D_R_payload_accessed"] is False
    assert dry["D_V_accessed"] is False
    assert dry["D_T_accessed"] is False
    assert dry["real_catalog_loader_call_count"] == 0
    assert dry["real_runner_publication_covered"] is False

    path = _ROOT / runner.DRY_RESULT_REPO_PATH
    result = _load(path)
    assert file_sha256(path) == _DRY_RESULT_SHA256
    assert path.stat().st_size == _DRY_RESULT_BYTE_COUNT
    assert result["result_fingerprint"] == _DRY_RESULT_FINGERPRINT
    _unsigned(result, fingerprint_field="result_fingerprint")


def _assert_runtime_and_dependency_bindings(
    closure: Mapping[str, object],
) -> dict[str, object]:
    signed = closure["runtime_implementation_binding"]
    assert isinstance(signed, dict)
    unsigned = _unsigned(
        signed,
        fingerprint_field="receipt_fingerprint",
    )
    current = runner._implementation_binding()
    assert unsigned == current
    assert signed["receipt_fingerprint"] == stable_fingerprint(current)
    assert unsigned["schema_version"] == runner.IMPLEMENTATION_SCHEMA
    assert unsigned["v4_implementation_receipt_fingerprint"] == (
        runner.V4_IMPLEMENTATION_RECEIPT_FINGERPRINT
    )

    v4_files = unsigned["v4_runtime_files"]
    v7_files = unsigned["v7_runtime_files"]
    all_files = unsigned["all_runtime_files"]
    assert isinstance(v4_files, dict)
    assert isinstance(v7_files, dict)
    assert isinstance(all_files, dict)
    assert len(v4_files) == 45
    assert len(v7_files) == 5
    assert len(all_files) == 50
    assert set(v4_files).isdisjoint(v7_files)
    assert all_files == {**v4_files, **v7_files}
    for repo_path, digest in all_files.items():
        assert file_sha256(_ROOT / repo_path) == digest

    dependency = closure["dependency_audit"]
    assert isinstance(dependency, dict)
    assert dependency["v4_runtime_file_count"] == 45
    assert dependency["v7_runtime_file_count"] == 5
    assert dependency["all_runtime_file_count"] == 50
    assert dependency["all_runtime_hashes_verified"] is True
    assert dependency["v4_runtime_fingerprint"] == (
        runner.V4_IMPLEMENTATION_RECEIPT_FINGERPRINT
    )
    assert dependency["v4_runtime_files"] == v4_files
    assert dependency["v7_runtime_files"] == v7_files
    assert dependency["all_runtime_files"] == all_files
    assert dependency["test_files"] == _repo_file_mapping(
        _IMPLEMENTATION_TEST_REPO_PATHS
    )
    return signed


def _assert_test_record(
    record: object,
    *,
    expected_files: Mapping[str, str],
    expected_deselected_count: int = 0,
) -> Mapping[str, object]:
    assert isinstance(record, dict)
    assert record["evidence_stage"] == "pre_signing"
    assert record["closure_receipt_present_during_execution"] is False
    assert record["exit_code"] == 0
    assert record["failed_count"] == 0
    assert isinstance(record["passed_count"], int)
    assert not isinstance(record["passed_count"], bool)
    assert record["passed_count"] >= 1
    assert isinstance(record["skipped_count"], int)
    assert record["skipped_count"] >= 0
    assert record["deselected_count"] == expected_deselected_count
    assert record["selected_count"] == (
        record["passed_count"] + record["skipped_count"]
    )
    assert record["collected_count"] == (
        record["selected_count"] + record["deselected_count"]
    )
    assert isinstance(record["command"], list)
    assert record["command"]
    assert "pytest" in " ".join(str(item) for item in record["command"])
    assert record["test_file_sha256"] == dict(expected_files)
    assert record["D_R_payload_accessed"] is False
    assert record["D_V_accessed"] is False
    assert record["D_T_accessed"] is False
    return record


def _assert_pre_signing_test_evidence(
    closure: Mapping[str, object],
) -> None:
    tests = closure["test_evidence"]
    assert isinstance(tests, dict)
    assert set(tests) == {
        "focused_tests",
        "full_regression",
        "real_runner_publication_tests",
        "gpu_temperature_wrapper_tests",
        "closure_static_test",
    }

    focused_files = _repo_file_mapping(_FOCUSED_TEST_REPO_PATHS)
    focused = _assert_test_record(
        tests["focused_tests"],
        expected_files=focused_files,
    )
    for repo_path in _FOCUSED_TEST_REPO_PATHS:
        assert repo_path in " ".join(
            str(item) for item in focused["command"]
        )

    all_tests = _all_test_file_mapping()
    full = _assert_test_record(
        tests["full_regression"],
        expected_files=all_tests,
        expected_deselected_count=1,
    )
    assert full["test_inventory_file_count"] == len(all_tests)
    assert full["test_inventory_fingerprint"] == stable_fingerprint(
        all_tests
    )
    assert full["pre_signing_excluded_test_files"] == [_SELF_REPO_PATH]
    assert full["pre_signing_deselected_real_payload_tests"] == [
        runner._PRE_SIGNING_REAL_PAYLOAD_TEST_NODE
    ]
    assert full["deselection_reason"] == (
        runner._PRE_SIGNING_REAL_PAYLOAD_DESELECTION_REASON
    )
    full_command = " ".join(str(item) for item in full["command"])
    assert "--ignore" in full_command
    assert _SELF_REPO_PATH in full_command
    assert "--deselect" in full_command
    assert runner._PRE_SIGNING_REAL_PAYLOAD_TEST_NODE in full_command

    publication = _assert_test_record(
        tests["real_runner_publication_tests"],
        expected_files=_repo_file_mapping(
            (_REAL_RUNNER_TEST_REPO_PATH,)
        ),
    )
    for field in (
        "completed_pass_publication_verified",
        "completed_nonpass_publication_verified",
        "execution_error_publication_verified",
        "strict_loader_verified",
        "signed_outer_to_unsigned_core_verified",
        "closure_failure_precedes_D_R_loader_verified",
        "authorization_failure_precedes_D_R_loader_verified",
        "authorization_verified_before_first_D_R_loader",
        "D_R_reconstruction_failure_publication_verified",
    ):
        assert publication[field] is True

    wrapper = _assert_test_record(
        tests["gpu_temperature_wrapper_tests"],
        expected_files=_repo_file_mapping(
            (_WRAPPER_TEST_REPO_PATH,)
        ),
    )
    assert wrapper["wrapper_tests_passed"] is True

    post_signing = tests["closure_static_test"]
    assert post_signing == {
        "repo_path": _SELF_REPO_PATH,
        "file_sha256": file_sha256(_ROOT / _SELF_REPO_PATH),
        "excluded_from_pre_signing_run": True,
        "post_signing_execution_required": True,
        "D_R_payload_access_allowed": False,
    }


def _assert_sync_and_temperature_bindings(
    closure: Mapping[str, object],
) -> None:
    sync = closure["sync_benchmark_binding"]
    assert isinstance(sync, dict)
    assert sync["repo_path"] == _SYNC_REPO_PATH
    assert sync["file_sha256"] == _SYNC_SHA256
    assert sync["result_fingerprint"] == _SYNC_FINGERPRINT
    assert sync["bounded_400_policy"] == "retain_strict_current_operator"
    assert sync["formal_800_authorized"] is False
    sync_path = _ROOT / _SYNC_REPO_PATH
    sync_payload = _load(sync_path)
    assert file_sha256(sync_path) == _SYNC_SHA256
    assert sync_payload["result_fingerprint"] == _SYNC_FINGERPRINT
    _unsigned(sync_payload, fingerprint_field="result_fingerprint")

    temperature = closure["gpu_temperature_control_evidence"]
    assert isinstance(temperature, dict)
    assert temperature["wrapper_repo_path"] == _WRAPPER_REPO_PATH
    assert temperature["wrapper_file_sha256"] == _WRAPPER_SHA256
    assert temperature["wrapper_binding_fingerprint"] == (
        stable_fingerprint(
            {
                "repo_path": _WRAPPER_REPO_PATH,
                "file_sha256": _WRAPPER_SHA256,
            }
        )
    )
    assert temperature["wrapper_test_repo_path"] == (
        _WRAPPER_TEST_REPO_PATH
    )
    assert temperature["wrapper_test_file_sha256"] == file_sha256(
        _ROOT / _WRAPPER_TEST_REPO_PATH
    )
    assert file_sha256(_ROOT / _WRAPPER_REPO_PATH) == _WRAPPER_SHA256
    assert temperature["gpu_index"] == 0
    assert temperature["pause_temperature_celsius"] == 82
    assert temperature["resume_temperature_celsius"] == 75
    assert temperature["tests_passed"] is True


def _assert_gate_and_authorization_boundary(
    closure: Mapping[str, object],
) -> None:
    gate = closure["gate_summary"]
    assert isinstance(gate, dict)
    for field in (
        "all_required_checks_pass",
        "dry_two_process_byte_replay_verified",
        "all_50_runtime_hashes_verified",
        "real_runner_three_outcomes_verified",
        "strict_real_runner_loader_verified",
        "signed_outer_to_unsigned_core_verified",
        "closure_failure_precedes_D_R_loader_verified",
        "authorization_failure_precedes_D_R_loader_verified",
        "authorization_verified_before_first_D_R_loader",
        "D_R_reconstruction_failure_publication_verified",
    ):
        assert gate[field] is True

    boundary = closure["boundary"]
    assert isinstance(boundary, dict)
    for field in (
        "D_R_payload_accessed",
        "real_D_R_bounded_authorized",
        "D_V_accessed",
        "D_T_accessed",
        "calibration_performed",
        "detection_performance_evaluated",
        "formal_800_authorized",
        "full_CURE_authorized",
        "other_detector_integration_authorized",
        "real_run_authorization_created",
    ):
        assert boundary[field] is False

    eligibility = closure["authorization_eligibility"]
    assert isinstance(eligibility, dict)
    assert eligibility["single_real_D_R_run_eligible"] is True
    assert eligibility["directly_authorizes_real_D_R_run"] is False
    assert eligibility["separate_signed_authorization_required"] is True
    assert eligibility["formal_800_authorized"] is False


def test_signed_bounded_implementation_closure_is_complete_and_static() -> None:
    assert _CLOSURE.is_file(), (
        "the bounded implementation closure has not been created; "
        "this test is intentionally post-signing"
    )
    closure = _load(_CLOSURE)
    _unsigned(closure, fingerprint_field="receipt_fingerprint")
    assert closure["schema_version"] == runner.IMPLEMENTATION_CLOSURE_SCHEMA
    assert closure["method_id"] == "cr_lvec_v7"
    assert closure["phase_status"] == "FROZEN_BOUNDED_IMPLEMENTATION_PASS"
    assert closure["decision"] == (
        "CR_LVEC_V7_BOUNDED_IMPLEMENTATION_GATE_PASS"
    )

    _assert_protocol_bindings(closure)
    _assert_dry_replay_binding(closure)
    signed_runtime = _assert_runtime_and_dependency_bindings(closure)
    _assert_pre_signing_test_evidence(closure)
    _assert_sync_and_temperature_bindings(closure)
    _assert_gate_and_authorization_boundary(closure)

    config = runner._load_config(_ROOT / runner.CONFIG_REPO_PATH)
    current_runtime = runner._implementation_binding()
    loaded, path, loaded_runtime = runner._load_implementation_closure(
        config,
        current_runtime,
    )
    assert loaded == closure
    assert path == _CLOSURE
    assert loaded_runtime == signed_runtime
