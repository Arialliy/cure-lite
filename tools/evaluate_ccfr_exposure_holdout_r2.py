#!/usr/bin/env python3
"""Run the CCFR-v11 holdout with a pre-attempt serialization correction.

Revision r1 remains byte-frozen.  It never wrote an attempt or entered
optimization because one diagnostic exposure map used integer JSON keys.
This additive runner changes only that map's key representation from
``int(0..15)`` to decimal strings.  All model, data, optimization, gate, and
report-only comparator computations are delegated to the frozen r1 module.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
from typing import Sequence
import xml.etree.ElementTree as ET

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import (  # noqa: E402
    file_sha256,
    stable_fingerprint,
)
from tools import evaluate_ccfr_exposure_holdout as r1  # noqa: E402


SCHEMA_VERSION = "cure-lite-ccfr-v11-exposure-holdout-result-r2"
ATTEMPT_SCHEMA_VERSION = (
    "cure-lite-ccfr-v11-exposure-holdout-attempt-r2"
)
METHOD_ID = r1.METHOD_ID
STAGE_ID = r1.STAGE_ID
RUNNER_REVISION = "r2"
CORRECTION_ID = "preattempt_int_key_diagnostic_map_canonicalization_r2"
SCIENTIFIC_ATTEMPT_ORDINAL = 1
EXPECTED_FACTUAL_STATE_INDICES = tuple(range(16))
EXPECTED_FACTUAL_EXPOSURES_PER_STATE = 100

_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conditioned_feature_release_v11"
)
_FAILURE_RECEIPT = (
    _PROTOCOL / "exposure_holdout_r1_pre_attempt_failure_receipt.json"
)
_PRE_RUN_V1_INVALIDATION = (
    _PROTOCOL / "exposure_holdout_r2_pre_run_v1_invalidation_receipt.json"
)
_CORRECTION_CLOSURE = (
    _PROTOCOL / "exposure_holdout_implementation_closure_r2_v2.json"
)
_PRE_RUN_RECEIPT = (
    _PROTOCOL / "exposure_holdout_r2_pre_run_verification_receipt_v2.json"
)
_CANONICAL_ATTEMPT = _PROTOCOL / "exposure_holdout_attempt_r2.json"
_CANONICAL_RESULT = _PROTOCOL / "exposure_holdout_result_r2.json"
_CANONICAL_COMPLETE = (
    _PROTOCOL / "exposure_holdout_result_r2.COMPLETE.sha256"
)
_TARGETED_REPORT = (
    _PROTOCOL / "exposure_holdout_r2_targeted_pre_run_v2.junit.xml"
)
_BROAD_REPORT = (
    _PROTOCOL / "exposure_holdout_r2_broad_pre_run_v2.junit.xml"
)
_PRE_RUN_SELF_TEST_NODEID = (
    "tests_v11/test_ccfr_exposure_holdout_r2.py::"
    "test_pre_run_receipt_authorizes_one_r2_attempt"
)
_PRE_RUN_SELECTION = (
    "not pre_run_receipt_authorizes_one_r2_attempt"
)
_TARGETED_COMMAND = [
    "/home/md0/ly/MSHNet/.venv/bin/python",
    "-m",
    "pytest",
    "-q",
    "tests_v11/test_ccfr_exposure_holdout_r2.py",
    "-k",
    _PRE_RUN_SELECTION,
    "--junitxml=protocols/IRSTD-1K/"
    "coverage_conditioned_feature_release_v11/"
    "exposure_holdout_r2_targeted_pre_run_v2.junit.xml",
]
_BROAD_COMMAND = [
    "/home/md0/ly/MSHNet/.venv/bin/python",
    "-m",
    "pytest",
    "-q",
    "tests_v8",
    "tests_v9",
    "tests_v10",
    "tests_v11",
    "-k",
    _PRE_RUN_SELECTION,
    "--junitxml=protocols/IRSTD-1K/"
    "coverage_conditioned_feature_release_v11/"
    "exposure_holdout_r2_broad_pre_run_v2.junit.xml",
]
_R1_AUTHORITY_PATHS = (
    _PROTOCOL / "exposure_holdout_attempt_r1.json",
    _PROTOCOL / "exposure_holdout_result_r1.json",
    _PROTOCOL / "exposure_holdout_result_r1.COMPLETE.sha256",
)
_R2_SOURCE_PATHS = tuple(r1._SOURCE_PATHS) + (
    "tests_v11/test_ccfr_exposure_holdout_r2.py",
    "tools/evaluate_ccfr_exposure_holdout_r2.py",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ACTIVE_AUTHORITY_TOKEN: object | None = None


def _repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(_ROOT.resolve()))


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _verify_fingerprint(
    value: dict[str, object],
    *,
    field: str,
    name: str,
) -> str:
    unsigned = dict(value)
    observed = unsigned.pop(field, None)
    if (
        not isinstance(observed, str)
        or not _SHA256_PATTERN.fullmatch(observed)
        or stable_fingerprint(unsigned) != observed
    ):
        raise RuntimeError(f"{name} fingerprint differs")
    return observed


def _verify_bound_file(binding: object, *, name: str) -> None:
    if not isinstance(binding, dict):
        raise TypeError(f"{name} binding must be an object")
    repo_path = binding.get("repo_path")
    expected_hash = binding.get("file_sha256")
    if (
        not isinstance(repo_path, str)
        or not isinstance(expected_hash, str)
        or not _SHA256_PATTERN.fullmatch(expected_hash)
    ):
        raise RuntimeError(f"{name} binding is invalid")
    path = _ROOT / repo_path
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise RuntimeError(f"{name} bound file differs")


def _load_failure_receipt() -> dict[str, object]:
    receipt = _load_object(
        _FAILURE_RECEIPT,
        name="CCFR holdout r1 pre-attempt failure receipt",
    )
    fingerprint = _verify_fingerprint(
        receipt,
        field="failure_receipt_fingerprint",
        name="CCFR holdout r1 pre-attempt failure receipt",
    )
    if receipt.get("schema_version") != (
        "cure-lite-ccfr-v11-holdout-r1-pre-attempt-failure-v1"
    ) or receipt.get("method_id") != METHOD_ID or receipt.get(
        "stage_id"
    ) != STAGE_ID:
        raise RuntimeError("CCFR holdout r1 failure identity differs")
    if receipt.get("status") != (
        "SEALED_AFTER_R1_PRE_ATTEMPT_FAILURE_BEFORE_R2_IMPLEMENTATION"
    ):
        raise RuntimeError("CCFR holdout r1 failure status differs")
    scientific_state = receipt.get("scientific_state")
    if not isinstance(scientific_state, dict) or scientific_state != {
        "r1_scientific_attempt_consumed": False,
        "r1_attempt_receipt_written": False,
        "r1_holdout_optimizer_steps": 0,
        "ccfr_training_entered": False,
        "v8_comparator_training_entered": False,
        "holdout_model_outputs_observed": False,
        "holdout_performance_metrics_observed": False,
        "candidate_selection_performed": False,
    }:
        raise RuntimeError("CCFR holdout r1 scientific state differs")
    root_cause = receipt.get("root_cause")
    if (
        not isinstance(root_cause, dict)
        or root_cause.get("unique_non_string_mapping_path")
        != (
            "prerequisites.holdout_contract."
            "factual_exposures_per_state"
        )
        or root_cause.get("observed_keys")
        != list(EXPECTED_FACTUAL_STATE_INDICES)
        or root_cause.get("observed_value_per_key")
        != EXPECTED_FACTUAL_EXPOSURES_PER_STATE
        or root_cause.get("failure_precedes_attempt_write") is not True
        or root_cause.get("failure_precedes_training") is not True
        or root_cause.get("global_fingerprint_relaxation_allowed")
        is not False
    ):
        raise RuntimeError("CCFR holdout r1 root-cause binding differs")
    authorized = receipt.get("authorized_correction")
    if authorized != {
        "runner_revision": "r2",
        "scientific_attempt_number": 1,
        "only_allowed_value_change": (
            "convert factual_exposures_per_state keys 0..15 to decimal "
            "strings while preserving all counts"
        ),
        "recursive_or_generic_key_coercion_allowed": False,
        "model_change_allowed": False,
        "input_change_allowed": False,
        "schedule_change_allowed": False,
        "optimizer_change_allowed": False,
        "threshold_change_allowed": False,
        "decision_change_allowed": False,
        "automatic_retry_allowed": False,
    }:
        raise RuntimeError("CCFR holdout r1 authorized correction differs")
    r1_artifacts = receipt.get("r1_authority_artifacts")
    if r1_artifacts != {
        "attempt_repo_path": _repo_path(_R1_AUTHORITY_PATHS[0]),
        "attempt_absent": True,
        "result_repo_path": _repo_path(_R1_AUTHORITY_PATHS[1]),
        "result_absent": True,
        "complete_repo_path": _repo_path(_R1_AUTHORITY_PATHS[2]),
        "complete_absent": True,
    }:
        raise RuntimeError("CCFR r1 authority artifact receipt differs")
    launch_events = receipt.get("launch_events")
    if not isinstance(launch_events, list) or len(launch_events) != 2:
        raise RuntimeError("CCFR r1 launch event count differs")
    expected_launch_states = (
        ("argument_validation_only", 2, False),
        ("canonical_pre_attempt_construction", 1, True),
    )
    for event, expected in zip(launch_events, expected_launch_states):
        event_is_exact = isinstance(event, dict) and (
            event.get("event"),
            event.get("exit_code"),
            event.get("canonical_output_supplied"),
        ) == expected and all(
            event.get(field) is False
            for field in (
                "attempt_receipt_written",
                "training_entered",
                "scientific_attempt_consumed",
            )
        )
        if not event_is_exact:
            raise RuntimeError("CCFR r1 launch event state differs")
    for path in _R1_AUTHORITY_PATHS:
        if path.exists():
            raise RuntimeError("CCFR r1 authority artifact must remain absent")
    upstream = receipt.get("frozen_upstream_bindings")
    if not isinstance(upstream, dict):
        raise TypeError("CCFR holdout r1 upstream bindings must be an object")
    for name, binding in upstream.items():
        _verify_bound_file(binding, name=f"r1 upstream {name}")
    return {
        "repo_path": _repo_path(_FAILURE_RECEIPT),
        "file_sha256": file_sha256(_FAILURE_RECEIPT),
        "failure_receipt_fingerprint": fingerprint,
        "status": receipt["status"],
        "frozen_upstream_bindings": upstream,
    }


def _validate_source_bindings(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TypeError("CCFR r2 source_bindings must be an object")
    if set(value) != set(_R2_SOURCE_PATHS):
        raise RuntimeError("CCFR r2 exact source binding set differs")
    bindings: dict[str, str] = {}
    root = _ROOT.resolve()
    for repo_path in _R2_SOURCE_PATHS:
        expected_hash = value.get(repo_path)
        relative = Path(repo_path)
        if (
            not isinstance(expected_hash, str)
            or not _SHA256_PATTERN.fullmatch(expected_hash)
            or not repo_path
            or "\\" in repo_path
            or relative.is_absolute()
            or relative.as_posix() != repo_path
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise RuntimeError(
                f"CCFR r2 source path/hash is invalid: {repo_path}"
            )
        resolved = (_ROOT / relative).resolve()
        if (
            not resolved.is_relative_to(root)
            or not resolved.is_file()
            or file_sha256(resolved) != expected_hash
        ):
            raise RuntimeError(
                f"CCFR r2 bound source differs: {repo_path}"
            )
        bindings[repo_path] = expected_hash
    return bindings


def _load_correction_closure(
    failure_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    if failure_receipt is None:
        failure_receipt = _load_failure_receipt()
    receipt = _load_object(
        _CORRECTION_CLOSURE,
        name="CCFR holdout r2 implementation closure",
    )
    fingerprint = _verify_fingerprint(
        receipt,
        field="closure_fingerprint",
        name="CCFR holdout r2 implementation closure",
    )
    exact = {
        "schema_version": (
            "cure-lite-ccfr-v11-exposure-holdout-r2-closure-v2"
        ),
        "method_id": METHOD_ID,
        "stage_id": STAGE_ID,
        "correction_id": CORRECTION_ID,
        "status": "FROZEN_BEFORE_R2_ATTEMPT",
        "correction_scope": "PRE_ATTEMPT_SERIALIZATION_ONLY",
        "evidence_scope_clarification": {
            "r1_scientific_state_scope": (
                "listed_r1_launch_events_only"
            ),
            "launch_events_are_sealed_statements_not_raw_log_artifacts": (
                True
            ),
            "root_cause_independently_reproduced_by_tests": True,
            "preformal_wiring_smoke_disclosure_bound": True,
            "no_full_400_result_or_gate_metric_observed": True,
        },
        "canonicalization_contract": {
            "exact_path": (
                "prerequisites.holdout_contract."
                "factual_exposures_per_state"
            ),
            "accepted_input_keys": "exact_int_range_0_through_15",
            "accepted_value": "exact_int_100_for_each_state",
            "output_key_rule": "decimal_str_index_in_numeric_order",
            "values_changed": False,
            "cardinality_changed": False,
            "aggregate_changed": False,
            "recursive_or_generic_coercion_allowed": False,
            "json_round_trip_identity_required": True,
            "fingerprintability_required": True,
        },
        "scientific_contract_invariance": {
            "model": True,
            "data_generator": True,
            "pair_population": True,
            "features": True,
            "geometry": True,
            "schedule": True,
            "loss": True,
            "optimizer": True,
            "seed": True,
            "update_count": True,
            "thresholds": True,
            "decision": True,
            "comparator_report_only": True,
            "execution_boundary": True,
        },
        "r2_canonical_artifacts": {
            "attempt_repo_path": _repo_path(_CANONICAL_ATTEMPT),
            "result_repo_path": _repo_path(_CANONICAL_RESULT),
            "complete_repo_path": _repo_path(_CANONICAL_COMPLETE),
            "runner_revision": RUNNER_REVISION,
            "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
            "automatic_retry_allowed": False,
        },
        "execution_boundary": {
            "dataset_access_allowed": False,
            "D_R_access_allowed": False,
            "D_V_access_allowed": False,
            "D_T_access_allowed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_detector_authorized": False,
        },
        "attempt_budget": {
            "design_confirmation_budget": 1,
            "r1_scientific_attempts_consumed": 0,
            "r2_scientific_attempt_ordinal": 1,
            "remaining_before_r2_attempt": 1,
            "remaining_after_r2_attempt_write": 0,
            "r2_is_retry": False,
            "rerun_after_pass_fail_or_exception_allowed": False,
        },
        "hash_chain_chronology": [
            "frozen_design_receipt",
            "development_config_attempt_result_complete",
            "r1_pre_attempt_failure_receipt",
            "r2_pre_run_v1_invalidation_receipt",
            "r2_implementation_closure_v2",
            "r2_pre_run_verification_v2",
            "r2_attempt_create_only_consumes_budget",
            "r2_result_pass_or_fail_is_authoritative",
            "r2_complete",
        ],
    }
    for field, expected in exact.items():
        if receipt.get(field) != expected:
            raise RuntimeError(f"CCFR r2 closure differs: {field}")
    expected_top = set(exact) | {
        "failure_receipt_binding",
        "pre_run_v1_invalidation_binding",
        "frozen_authority_bindings",
        "source_bindings",
        "closure_fingerprint",
    }
    if set(receipt) != expected_top:
        raise RuntimeError("CCFR r2 closure top-level fields differ")
    failure_binding = receipt.get("failure_receipt_binding")
    _verify_bound_file(
        failure_binding,
        name="r2 failure receipt",
    )
    invalidation_binding = receipt.get(
        "pre_run_v1_invalidation_binding"
    )
    _verify_bound_file(
        invalidation_binding,
        name="r2 pre-run v1 invalidation receipt",
    )
    invalidation = _load_object(
        _PRE_RUN_V1_INVALIDATION,
        name="CCFR r2 pre-run v1 invalidation receipt",
    )
    invalidation_fingerprint = _verify_fingerprint(
        invalidation,
        field="invalidation_fingerprint",
        name="CCFR r2 pre-run v1 invalidation receipt",
    )
    if (
        invalidation.get("status")
        != "PRE_RUN_V1_INVALIDATED_BEFORE_ATTEMPT"
        or not isinstance(invalidation_binding, dict)
        or invalidation_binding != {
            "repo_path": _repo_path(_PRE_RUN_V1_INVALIDATION),
            "file_sha256": file_sha256(_PRE_RUN_V1_INVALIDATION),
            "invalidation_fingerprint": invalidation_fingerprint,
        }
        or invalidation.get("scientific_state")
        != {
            "attempt_receipt_written": False,
            "scientific_attempts_consumed": 0,
            "optimizer_steps": 0,
            "model_outputs_observed": False,
            "holdout_gate_metrics_observed": False,
            "remaining_scientific_attempts": 1,
        }
    ):
        raise RuntimeError("CCFR r2 pre-run v1 invalidation differs")
    authority = receipt.get("frozen_authority_bindings")
    if authority != failure_receipt["frozen_upstream_bindings"]:
        raise RuntimeError("CCFR r2 frozen authority bindings differ")
    for name, binding in authority.items():
        _verify_bound_file(binding, name=f"r2 authority {name}")
    bindings = _validate_source_bindings(receipt.get("source_bindings"))
    return {
        "repo_path": _repo_path(_CORRECTION_CLOSURE),
        "file_sha256": file_sha256(_CORRECTION_CLOSURE),
        "closure_fingerprint": fingerprint,
        "correction_id": receipt["correction_id"],
        "failure_receipt_binding": failure_binding,
        "pre_run_v1_invalidation_binding": invalidation_binding,
        "source_bindings": bindings,
    }


def _junit_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(
        root.findall("testsuite")
    )
    if not suites:
        raise RuntimeError("CCFR r2 JUnit report has no test suite")
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(
        int(suite.attrib.get("failures", "0")) for suite in suites
    )
    errors = sum(
        int(suite.attrib.get("errors", "0")) for suite in suites
    )
    skipped = sum(
        int(suite.attrib.get("skipped", "0")) for suite in suites
    )
    return {
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _validate_pre_run_test_evidence(
    value: object,
    *,
    expected_command: list[str],
    expected_report: Path,
    expected_scope: list[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("CCFR r2 test evidence must be an object")
    expected_top = {
        "command",
        "execution_cwd",
        "selection_scope",
        "exit_code",
        "outcome",
        "deselected",
        "junit",
        "summary",
    }
    if set(value) != expected_top:
        raise RuntimeError("CCFR r2 test evidence fields differ")
    if (
        value.get("command") != expected_command
        or value.get("execution_cwd") != "/home/md0/ly/cure_lite"
        or value.get("selection_scope") != expected_scope
        or value.get("exit_code") != 0
        or value.get("outcome") != "PASS"
        or value.get("deselected") != 1
    ):
        raise RuntimeError("CCFR r2 test evidence contract differs")
    junit = value.get("junit")
    if junit != {
        "repo_path": _repo_path(expected_report),
        "file_sha256": (
            junit.get("file_sha256")
            if isinstance(junit, dict)
            else None
        ),
    }:
        raise RuntimeError("CCFR r2 JUnit binding shape differs")
    assert isinstance(junit, dict)
    digest = junit.get("file_sha256")
    if (
        not isinstance(digest, str)
        or not _SHA256_PATTERN.fullmatch(digest)
        or not expected_report.is_file()
        or file_sha256(expected_report) != digest
    ):
        raise RuntimeError("CCFR r2 JUnit report binding differs")
    observed = _junit_summary(expected_report)
    if value.get("summary") != observed:
        raise RuntimeError("CCFR r2 JUnit summary differs")
    if (
        observed["tests"] <= 0
        or observed["failures"] != 0
        or observed["errors"] != 0
    ):
        raise RuntimeError("CCFR r2 pre-run tests did not pass")
    return value


def _load_pre_run_receipt(
    closure: dict[str, object],
    *,
    preauthorization_fingerprint: str,
) -> dict[str, object]:
    receipt = _load_object(
        _PRE_RUN_RECEIPT,
        name="CCFR holdout r2 pre-run verification receipt",
    )
    fingerprint = _verify_fingerprint(
        receipt,
        field="pre_run_fingerprint",
        name="CCFR holdout r2 pre-run verification receipt",
    )
    exact = {
        "schema_version": (
            "cure-lite-ccfr-v11-exposure-holdout-r2-pre-run-v2"
        ),
        "method_id": METHOD_ID,
        "stage_id": STAGE_ID,
        "status": "R2_SINGLE_ATTEMPT_AUTHORIZED",
        "runner_revision": RUNNER_REVISION,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "r2_is_retry": False,
        "attempt_budget": {
            "design_confirmation_budget": 1,
            "r1_scientific_attempts_consumed": 0,
            "r2_scientific_attempt_ordinal": 1,
            "remaining_before_attempt_write": 1,
            "remaining_after_attempt_write": 0,
            "pass_fail_or_exception_consumes_attempt": True,
            "rerun_allowed": False,
        },
        "authorization_scope": {
            "dataset_free_holdout_only": True,
            "PASS_authorizes_only_real_D_R_bounded_validation": True,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_detector_authorized": False,
        },
        "preauthorization_prerequisites_fingerprint": (
            preauthorization_fingerprint
        ),
    }
    for field, expected in exact.items():
        if receipt.get(field) != expected:
            raise RuntimeError(f"CCFR r2 pre-run differs: {field}")
    expected_top = set(exact) | {
        "closure_binding",
        "failure_receipt_binding",
        "artifact_absence_at_freeze",
        "verification",
        "pre_run_fingerprint",
    }
    if set(receipt) != expected_top:
        raise RuntimeError("CCFR r2 pre-run top-level fields differ")
    if receipt.get("closure_binding") != {
        "repo_path": closure["repo_path"],
        "file_sha256": closure["file_sha256"],
        "closure_fingerprint": closure["closure_fingerprint"],
    }:
        raise RuntimeError("CCFR r2 pre-run closure binding differs")
    if receipt.get("failure_receipt_binding") != closure[
        "failure_receipt_binding"
    ]:
        raise RuntimeError("CCFR r2 pre-run failure binding differs")
    artifacts = receipt.get("artifact_absence_at_freeze")
    expected_entries = [
        {"repo_path": _repo_path(path), "exists": False}
        for path in (
            *_R1_AUTHORITY_PATHS,
            _CANONICAL_ATTEMPT,
            _CANONICAL_RESULT,
            _CANONICAL_COMPLETE,
        )
    ]
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("entries") != expected_entries
        or artifacts.get("all_absent") is not True
    ):
        raise RuntimeError("CCFR r2 pre-run artifact state differs")
    verification = receipt.get("verification")
    if not isinstance(verification, dict) or set(verification) != {
        "authorization_test_scope",
        "targeted",
        "broad",
    }:
        raise RuntimeError("CCFR r2 pre-run verification differs")
    if verification.get("authorization_test_scope") != {
        "only_excluded_nodeid": _PRE_RUN_SELF_TEST_NODEID,
        "exclusion_expression": _PRE_RUN_SELECTION,
        "exclusion_reason": (
            "pre_run_receipt_cannot_self_authorize_its_own_validation"
        ),
        "receipt_dependent_test_claimed_as_preauthorization_evidence": (
            False
        ),
    }:
        raise RuntimeError("CCFR r2 authorization test scope differs")
    _validate_pre_run_test_evidence(
        verification.get("targeted"),
        expected_command=_TARGETED_COMMAND,
        expected_report=_TARGETED_REPORT,
        expected_scope=[
            "tests_v11/test_ccfr_exposure_holdout_r2.py"
        ],
    )
    _validate_pre_run_test_evidence(
        verification.get("broad"),
        expected_command=_BROAD_COMMAND,
        expected_report=_BROAD_REPORT,
        expected_scope=["tests_v8", "tests_v9", "tests_v10", "tests_v11"],
    )
    return {
        "repo_path": _repo_path(_PRE_RUN_RECEIPT),
        "file_sha256": file_sha256(_PRE_RUN_RECEIPT),
        "pre_run_fingerprint": fingerprint,
        "status": receipt["status"],
    }


def _non_string_mapping_paths(
    value: object,
    *,
    path: str = "$",
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                paths.append(path)
            paths.extend(
                _non_string_mapping_paths(item, path=f"{path}.{key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(
                _non_string_mapping_paths(
                    item,
                    path=f"{path}[{index}]",
                )
            )
    return paths


def _canonicalize_holdout_contract(
    raw_contract: dict[str, object],
) -> dict[str, object]:
    """Correct exactly one diagnostic map and reject every other shape."""

    contract = copy.deepcopy(raw_contract)
    raw = contract.get("factual_exposures_per_state")
    if not isinstance(raw, dict):
        raise TypeError("factual exposure diagnostic must be an object")
    if (
        len(raw) != len(EXPECTED_FACTUAL_STATE_INDICES)
        or any(type(key) is not int for key in raw)
        or set(raw) != set(EXPECTED_FACTUAL_STATE_INDICES)
        or any(
            type(value) is not int
            or value != EXPECTED_FACTUAL_EXPOSURES_PER_STATE
            for value in raw.values()
        )
        or sum(raw.values()) != 1600
    ):
        raise RuntimeError("factual exposure diagnostic shape differs")
    corrected = {
        str(index): raw[index]
        for index in EXPECTED_FACTUAL_STATE_INDICES
    }
    contract["factual_exposures_per_state"] = corrected
    if (
        len(corrected) != 16
        or sum(corrected.values()) != 1600
        or _non_string_mapping_paths(contract)
        or json.loads(
            json.dumps(
                contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        != contract
    ):
        raise RuntimeError("corrected holdout contract is not canonical")
    stable_fingerprint(contract)
    return contract


def _load_pre_authorization_prerequisites() -> dict[str, object]:
    """Load every frozen prerequisite that precedes r2 authorization."""

    runtime_boundary = r1._runtime_import_boundary()
    design_receipt = r1._load_design_receipt()
    development_pass = r1._load_development_pass()
    development_protocol = development_pass.get("protocol_binding")
    if not isinstance(development_protocol, dict):
        raise TypeError("CCFR development protocol binding must be an object")
    expected_holdout_binding = {
        name: design_receipt[name]
        for name in (
            "repo_path",
            "file_sha256",
            "receipt_fingerprint",
            "status",
            "design_seed",
        )
    }
    if development_protocol.get("pre_frozen_holdout_binding") != (
        expected_holdout_binding
    ):
        raise RuntimeError("development does not bind the frozen holdout")
    development_sources = development_protocol.get("source_bindings")
    if (
        not isinstance(development_sources, dict)
        or development_sources.get(design_receipt["repo_path"])
        != design_receipt["file_sha256"]
    ):
        raise RuntimeError("development does not bind holdout source")

    raw_contract = r1._holdout_contract()
    if raw_contract["implementation_fingerprints"] != (
        design_receipt["implementation_fingerprints"]
    ):
        raise RuntimeError("holdout implementation differs from design")
    holdout_contract = _canonicalize_holdout_contract(raw_contract)
    failure_receipt = _load_failure_receipt()
    closure = _load_correction_closure(failure_receipt)
    if closure["failure_receipt_binding"] != {
        "repo_path": failure_receipt["repo_path"],
        "file_sha256": failure_receipt["file_sha256"],
        "failure_receipt_fingerprint": failure_receipt[
            "failure_receipt_fingerprint"
        ],
    }:
        raise RuntimeError("CCFR r2 failure receipt binding differs")
    design_sources = design_receipt.get("source_bindings")
    if not isinstance(design_sources, dict):
        raise TypeError("CCFR design source bindings must be an object")
    if any(
        closure["source_bindings"].get(path) != digest
        for path, digest in design_sources.items()
    ) or set(closure["source_bindings"]) - set(design_sources) != {
        "tests_v11/test_ccfr_exposure_holdout_r2.py",
        "tools/evaluate_ccfr_exposure_holdout_r2.py",
    }:
        raise RuntimeError("CCFR r2 effective source override differs")
    runtime_boundary["local_source_closure"] = (
        r1._runtime_source_closure(closure["source_bindings"])
    )
    prerequisites: dict[str, object] = {
        "design_receipt": design_receipt,
        "development_pass": development_pass,
        "holdout_contract": holdout_contract,
        "runtime_import_boundary": runtime_boundary,
        "r1_pre_attempt_failure": failure_receipt,
        "r2_implementation_closure": {
            key: closure[key]
            for key in (
                "repo_path",
                "file_sha256",
                "closure_fingerprint",
                "correction_id",
                "failure_receipt_binding",
            )
        },
        "source_bindings": closure["source_bindings"],
    }
    if _non_string_mapping_paths(prerequisites):
        raise RuntimeError("CCFR r2 prerequisites contain non-string keys")
    stable_fingerprint(prerequisites)
    return prerequisites


def _load_prerequisites() -> dict[str, object]:
    prerequisites = _load_pre_authorization_prerequisites()
    closure = prerequisites["r2_implementation_closure"]
    prerequisites["r2_pre_run_verification"] = _load_pre_run_receipt(
        closure,
        preauthorization_fingerprint=stable_fingerprint(prerequisites),
    )
    if _non_string_mapping_paths(prerequisites):
        raise RuntimeError("CCFR r2 prerequisites contain non-string keys")
    stable_fingerprint(prerequisites)
    return prerequisites


def _attempt_payload(
    prerequisites: dict[str, object],
) -> dict[str, object]:
    attempt: dict[str, object] = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "stage_id": STAGE_ID,
        "runner_revision": RUNNER_REVISION,
        "correction_id": CORRECTION_ID,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "r1_pre_attempt_failure_consumed_scientific_attempt": False,
        "status": (
            "SINGLE_SCIENTIFIC_ATTEMPT_CONSUMED_BEFORE_OPTIMIZATION"
        ),
        "prerequisites": prerequisites,
        "canonical_artifacts": {
            "attempt_repo_path": _repo_path(_CANONICAL_ATTEMPT),
            "result_repo_path": _repo_path(_CANONICAL_RESULT),
            "complete_repo_path": _repo_path(_CANONICAL_COMPLETE),
        },
        "execution": {
            "updates": r1.UPDATE_COUNT,
            "device": "cpu",
            "torch_threads": 2,
            "automatic_retry_allowed": False,
            "dataset_access_allowed": False,
            "D_R_access_allowed": False,
            "D_V_access_allowed": False,
            "D_T_access_allowed": False,
        },
    }
    if _non_string_mapping_paths(attempt):
        raise RuntimeError("CCFR r2 attempt contains non-string keys")
    attempt["attempt_fingerprint"] = stable_fingerprint(attempt)
    if json.loads(
        json.dumps(
            attempt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    ) != attempt:
        raise RuntimeError("CCFR r2 attempt JSON round trip differs")
    return attempt


def _write_json_create_only(
    path: Path,
    value: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _write_attempt_and_issue_authority(
    attempt: dict[str, object],
) -> object:
    """Write the sole attempt before issuing one process-local run token."""

    global _ACTIVE_AUTHORITY_TOKEN
    if _ACTIVE_AUTHORITY_TOKEN is not None:
        raise RuntimeError("CCFR r2 authority token is already active")
    _write_json_create_only(_CANONICAL_ATTEMPT, attempt)
    token = object()
    _ACTIVE_AUTHORITY_TOKEN = token
    return token


def _load_attempt(
    prerequisites: dict[str, object],
) -> dict[str, object]:
    attempt = _load_object(
        _CANONICAL_ATTEMPT,
        name="CCFR holdout r2 canonical attempt",
    )
    fingerprint = _verify_fingerprint(
        attempt,
        field="attempt_fingerprint",
        name="CCFR holdout r2 canonical attempt",
    )
    if attempt != _attempt_payload(prerequisites):
        raise RuntimeError("CCFR holdout r2 attempt differs")
    return {
        "repo_path": _repo_path(_CANONICAL_ATTEMPT),
        "file_sha256": file_sha256(_CANONICAL_ATTEMPT),
        "attempt_fingerprint": fingerprint,
        "runner_revision": RUNNER_REVISION,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
    }


def _assemble_r2_result(
    *,
    prerequisites: dict[str, object],
    attempt_binding: dict[str, object],
    ccfr: dict[str, object],
    comparator: dict[str, object],
    runtime: dict[str, object],
) -> dict[str, object]:
    result = r1._assemble_result(
        prerequisites=prerequisites,
        attempt_binding=attempt_binding,
        ccfr=ccfr,
        comparator=comparator,
        runtime=runtime,
    )
    frozen_assembler_fingerprint = result.pop("result_fingerprint")
    result["schema_version"] = SCHEMA_VERSION
    result["runner_revision"] = RUNNER_REVISION
    result["scientific_attempt_ordinal"] = SCIENTIFIC_ATTEMPT_ORDINAL
    result["serialization_correction"] = {
        "exact_path": (
            "prerequisites.holdout_contract."
            "factual_exposures_per_state"
        ),
        "input_keys": "int_0_through_15",
        "output_keys": "decimal_strings_0_through_15",
        "values_changed": False,
        "scientific_contract_changed": False,
        "correction_id": CORRECTION_ID,
        "frozen_r1_assembler_output_fingerprint_before_r2_annotations": (
            frozen_assembler_fingerprint
        ),
    }
    result["attempt_budget"] = {
        "design_confirmation_budget": 1,
        "r1_scientific_attempts_consumed": 0,
        "r2_scientific_attempts_consumed": 1,
        "remaining_after_r2_attempt_write": 0,
        "rerun_after_pass_fail_or_exception_allowed": False,
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def evaluate(
    *,
    _authority_token: object | None = None,
) -> dict[str, object]:
    global _ACTIVE_AUTHORITY_TOKEN
    if (
        _authority_token is None
        or _authority_token is not _ACTIVE_AUTHORITY_TOKEN
    ):
        raise RuntimeError(
            "CCFR r2 evaluation is available only through canonical main"
        )
    _ACTIVE_AUTHORITY_TOKEN = None
    prerequisites = _load_prerequisites()
    attempt_binding = _load_attempt(prerequisites)
    specs = r1.build_ccfr_holdout_pair_specs()
    schedule = r1.build_ccfr_holdout_schedule(specs)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        ccfr = r1._train_and_evaluate(
            objective_id=METHOD_ID,
            decoder_class=r1.CURELiteCoverageFeatureReleaseDecoder,
            specs=specs,
            schedule=schedule,
        )
        try:
            comparator = r1._train_and_evaluate(
                objective_id=r1.COMPARATOR_ID,
                decoder_class=r1.CURELiteConservativeFactorizedDecoder,
                specs=specs,
                schedule=schedule,
            )
        except Exception as error:  # frozen report-only boundary
            comparator = {
                "objective_id": r1.COMPARATOR_ID,
                "execution_status": "ERROR",
                "comparator_execution_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "all_pass": False,
            }
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)
    runtime = {
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "threads_before": previous_threads,
        "threads_during_confirmation": 2,
        "deterministic_algorithms_before": previous_deterministic,
        "deterministic_algorithms_during_confirmation": True,
        "threads_restored": torch.get_num_threads() == previous_threads,
        "deterministic_algorithms_restored": (
            torch.are_deterministic_algorithms_enabled()
            == previous_deterministic
        ),
    }
    return _assemble_r2_result(
        prerequisites=prerequisites,
        attempt_binding=attempt_binding,
        ccfr=ccfr,
        comparator=comparator,
        runtime=runtime,
    )


def _assert_fresh_canonical_artifacts(output: Path) -> None:
    if output.resolve() != _CANONICAL_RESULT.resolve():
        raise ValueError(
            "CCFR r2 output must use the frozen canonical path: "
            f"{_CANONICAL_RESULT}"
        )
    existing_r1 = [path for path in _R1_AUTHORITY_PATHS if path.exists()]
    if existing_r1:
        raise FileExistsError(
            f"CCFR r1 authority artifacts must remain absent: {existing_r1}"
        )
    existing_r2 = [
        path
        for path in (
            _CANONICAL_ATTEMPT,
            _CANONICAL_RESULT,
            _CANONICAL_COMPLETE,
        )
        if path.exists()
    ]
    if existing_r2:
        raise FileExistsError(
            "CCFR r2 single attempt is unavailable because an authority "
            f"artifact exists: {existing_r2}"
        )


def _write_complete_create_only(
    *,
    result: dict[str, object],
    attempt: dict[str, object],
    prerequisites: dict[str, object],
) -> str:
    disk_attempt = _load_object(
        _CANONICAL_ATTEMPT,
        name="CCFR r2 attempt before COMPLETE",
    )
    _verify_fingerprint(
        disk_attempt,
        field="attempt_fingerprint",
        name="CCFR r2 attempt before COMPLETE",
    )
    if disk_attempt != attempt:
        raise RuntimeError("CCFR r2 in-memory/disk attempt differs")
    disk_result = _load_object(
        _CANONICAL_RESULT,
        name="CCFR r2 result before COMPLETE",
    )
    _verify_fingerprint(
        disk_result,
        field="result_fingerprint",
        name="CCFR r2 result before COMPLETE",
    )
    if disk_result != result:
        raise RuntimeError("CCFR r2 in-memory/disk result differs")
    result_sha = file_sha256(_CANONICAL_RESULT)
    failure = prerequisites["r1_pre_attempt_failure"]
    closure = prerequisites["r2_implementation_closure"]
    pre_run = prerequisites["r2_pre_run_verification"]
    for path, binding, field, name in (
        (
            _FAILURE_RECEIPT,
            failure,
            "failure_receipt_fingerprint",
            "failure receipt before COMPLETE",
        ),
        (
            _CORRECTION_CLOSURE,
            closure,
            "closure_fingerprint",
            "implementation closure before COMPLETE",
        ),
        (
            _PRE_RUN_RECEIPT,
            pre_run,
            "pre_run_fingerprint",
            "pre-run receipt before COMPLETE",
        ),
    ):
        disk_receipt = _load_object(path, name=name)
        fingerprint = _verify_fingerprint(
            disk_receipt,
            field=field,
            name=name,
        )
        if (
            file_sha256(path) != binding["file_sha256"]
            or fingerprint != binding[field]
        ):
            raise RuntimeError(f"CCFR r2 {name} binding differs")
    payload = (
        f"{result_sha}  {_CANONICAL_RESULT.name}\n"
        f"attempt_sha256={file_sha256(_CANONICAL_ATTEMPT)}\n"
        f"attempt_fingerprint={attempt['attempt_fingerprint']}\n"
        f"failure_receipt_sha256={failure['file_sha256']}\n"
        "failure_receipt_fingerprint="
        f"{failure['failure_receipt_fingerprint']}\n"
        f"closure_sha256={closure['file_sha256']}\n"
        f"closure_fingerprint={closure['closure_fingerprint']}\n"
        f"pre_run_sha256={pre_run['file_sha256']}\n"
        f"pre_run_fingerprint={pre_run['pre_run_fingerprint']}\n"
        f"result_fingerprint={result['result_fingerprint']}\n"
    )
    with _CANONICAL_COMPLETE.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return result_sha


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _assert_fresh_canonical_artifacts(args.output)
    prerequisites = _load_prerequisites()
    attempt = _attempt_payload(prerequisites)
    authority_token = _write_attempt_and_issue_authority(attempt)
    result = evaluate(_authority_token=authority_token)
    _write_json_create_only(_CANONICAL_RESULT, result)
    result_sha = _write_complete_create_only(
        result=result,
        attempt=attempt,
        prerequisites=prerequisites,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "all_pass": result["all_pass"],
                "runner_revision": RUNNER_REVISION,
                "result_fingerprint": result["result_fingerprint"],
                "complete_sha256": result_sha,
                "output": str(_CANONICAL_RESULT),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["all_pass"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
