from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from tools import evaluate_ccfr_exposure_holdout as r1
from tools import evaluate_ccfr_exposure_holdout_r2 as r2


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conditioned_feature_release_v11"
)
FAILURE_RECEIPT = (
    PROTOCOL / "exposure_holdout_r1_pre_attempt_failure_receipt.json"
)
CLOSURE = r2._CORRECTION_CLOSURE
PRE_RUN = r2._PRE_RUN_RECEIPT
EVALUATOR = ROOT / "tools" / "evaluate_ccfr_exposure_holdout_r2.py"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_r1_failure_is_sealed_before_attempt_and_training() -> None:
    receipt = _load(FAILURE_RECEIPT)
    unsigned = dict(receipt)
    fingerprint = unsigned.pop("failure_receipt_fingerprint")

    assert stable_fingerprint(unsigned) == fingerprint
    assert receipt["scientific_state"] == {
        "r1_scientific_attempt_consumed": False,
        "r1_attempt_receipt_written": False,
        "r1_holdout_optimizer_steps": 0,
        "ccfr_training_entered": False,
        "v8_comparator_training_entered": False,
        "holdout_model_outputs_observed": False,
        "holdout_performance_metrics_observed": False,
        "candidate_selection_performed": False,
    }
    assert receipt["root_cause"]["observed_keys"] == list(range(16))
    assert receipt["root_cause"]["observed_value_per_key"] == 100
    for path in r2._R1_AUTHORITY_PATHS:
        assert not path.exists()


def test_r1_real_contract_fails_fingerprint_before_attempt_write() -> None:
    raw = r1._holdout_contract()
    exposure = raw["factual_exposures_per_state"]

    assert sorted(exposure) == list(range(16))
    assert set(exposure.values()) == {100}
    assert sum(exposure.values()) == 1600
    with pytest.raises(
        TypeError,
        match="fingerprint mapping keys must be strings",
    ):
        r1._attempt_payload({"holdout_contract": raw})
    for path in r2._R1_AUTHORITY_PATHS:
        assert not path.exists()


def test_targeted_canonicalization_changes_only_key_representation() -> None:
    raw = r1._holdout_contract()
    corrected = r2._canonicalize_holdout_contract(raw)
    corrected_exposure = corrected["factual_exposures_per_state"]

    assert list(corrected_exposure) == [str(index) for index in range(16)]
    assert set(corrected_exposure.values()) == {100}
    assert sum(corrected_exposure.values()) == 1600
    for key, value in raw.items():
        if key != "factual_exposures_per_state":
            assert corrected[key] == value
    assert raw["factual_exposures_per_state"] == {
        index: 100 for index in range(16)
    }
    assert r2._non_string_mapping_paths(corrected) == []
    assert json.loads(json.dumps(corrected, sort_keys=True)) == corrected
    assert len(stable_fingerprint(corrected)) == 64


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.pop(0),
        lambda value: value.__setitem__(16, 100),
        lambda value: (
            value.pop(0),
            value.__setitem__(False, 100),
        ),
        lambda value: value.__setitem__("0", value.pop(0)),
        lambda value: value.__setitem__(0, 99),
    ],
)
def test_targeted_canonicalization_rejects_every_other_shape(
    mutator: object,
) -> None:
    raw = r1._holdout_contract()
    changed = copy.deepcopy(raw)
    mutator(changed["factual_exposures_per_state"])

    with pytest.raises(RuntimeError, match="diagnostic shape differs"):
        r2._canonicalize_holdout_contract(changed)


def test_r2_delegates_all_scientific_computation_to_frozen_r1() -> None:
    assert r2.METHOD_ID == r1.METHOD_ID
    assert r2.STAGE_ID == r1.STAGE_ID
    source = EVALUATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "tools" in modules
    assert "cure_lite.coverage_feature_release_decoder" not in modules
    assert "cure_lite.conservative_factorized_decoder" not in modules
    assert "cure_lite.experiment" not in modules
    assert "datasets" not in modules
    assert "r1._train_and_evaluate" in source
    assert "r1._assemble_result" in source


def test_closure_binds_exact_effective_source_set_and_clean_import() -> None:
    closure = _load(CLOSURE)
    unsigned = dict(closure)
    fingerprint = unsigned.pop("closure_fingerprint")

    assert stable_fingerprint(unsigned) == fingerprint
    assert set(closure["source_bindings"]) == set(r2._R2_SOURCE_PATHS)
    assert r2._validate_source_bindings(
        closure["source_bindings"]
    ) == closure["source_bindings"]

    command = """
import json
from tools import evaluate_ccfr_exposure_holdout as r1
from tools import evaluate_ccfr_exposure_holdout_r2 as r2
closure = r2._load_correction_closure()
boundary = r1._runtime_import_boundary()
source = r1._runtime_source_closure(closure['source_bindings'])
print(json.dumps({'boundary': boundary, 'source': source}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["boundary"]["observed_forbidden_modules"] == []
    assert result["source"]["unbound_local_imports"] == []


def test_real_corrected_prerequisites_and_attempt_are_canonical() -> None:
    command = """
import json
from cure_lite.cache.schema import stable_fingerprint
from tools import evaluate_ccfr_exposure_holdout_r2 as r2
p = r2._load_pre_authorization_prerequisites()
a = r2._attempt_payload(p)
print(json.dumps({
    'bad_keys': r2._non_string_mapping_paths(p),
    'exposure': p['holdout_contract']['factual_exposures_per_state'],
    'prerequisites_fingerprint': stable_fingerprint(p),
    'runner_revision': a['runner_revision'],
    'scientific_attempt_ordinal': a['scientific_attempt_ordinal'],
    'r1_consumed': a[
        'r1_pre_attempt_failure_consumed_scientific_attempt'
    ],
    'round_trip_equal': json.loads(json.dumps(a, sort_keys=True)) == a,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["bad_keys"] == []
    assert result["exposure"] == {
        str(index): 100 for index in range(16)
    }
    assert len(result["prerequisites_fingerprint"]) == 64
    assert result["runner_revision"] == "r2"
    assert result["scientific_attempt_ordinal"] == 1
    assert result["r1_consumed"] is False
    assert result["round_trip_equal"] is True


def test_pre_run_receipt_authorizes_one_r2_attempt() -> None:
    receipt = _load(PRE_RUN)
    unsigned = dict(receipt)
    fingerprint = unsigned.pop("pre_run_fingerprint")

    assert stable_fingerprint(unsigned) == fingerprint
    command = """
import json
from cure_lite.cache.schema import stable_fingerprint
from tools import evaluate_ccfr_exposure_holdout_r2 as r2
pre = r2._load_pre_authorization_prerequisites()
closure = r2._load_correction_closure()
loaded = r2._load_pre_run_receipt(
    closure,
    preauthorization_fingerprint=stable_fingerprint(pre),
)
full = r2._load_prerequisites()
attempt = r2._attempt_payload(full)
print(json.dumps({
    'status': loaded['status'],
    'full_fingerprint': stable_fingerprint(full),
    'attempt_fingerprint': attempt['attempt_fingerprint'],
    'bad_keys': r2._non_string_mapping_paths(full),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "R2_SINGLE_ATTEMPT_AUTHORIZED"
    assert len(result["full_fingerprint"]) == 64
    assert len(result["attempt_fingerprint"]) == 64
    assert result["bad_keys"] == []
    verification = receipt["verification"]
    assert verification["authorization_test_scope"][
        "only_excluded_nodeid"
    ].endswith("test_pre_run_receipt_authorizes_one_r2_attempt")
    for name in ("targeted", "broad"):
        assert verification[name]["exit_code"] == 0
        assert verification[name]["summary"]["failures"] == 0
        assert verification[name]["summary"]["errors"] == 0
        report = ROOT / verification[name]["junit"]["repo_path"]
        assert file_sha256(report) == verification[name]["junit"][
            "file_sha256"
        ]


def test_canonical_artifact_guard_never_writes_r1_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r1_paths = tuple(tmp_path / f"r1-{index}" for index in range(3))
    r2_attempt = tmp_path / "r2-attempt.json"
    r2_result = tmp_path / "r2-result.json"
    r2_complete = tmp_path / "r2-result.COMPLETE.sha256"
    monkeypatch.setattr(r2, "_R1_AUTHORITY_PATHS", r1_paths)
    monkeypatch.setattr(r2, "_CANONICAL_ATTEMPT", r2_attempt)
    monkeypatch.setattr(r2, "_CANONICAL_RESULT", r2_result)
    monkeypatch.setattr(r2, "_CANONICAL_COMPLETE", r2_complete)

    r2._assert_fresh_canonical_artifacts(r2_result)
    with pytest.raises(ValueError, match="canonical path"):
        r2._assert_fresh_canonical_artifacts(tmp_path / "other.json")

    r2._write_json_create_only(r2_attempt, {"sentinel": True})
    with pytest.raises(FileExistsError):
        r2._write_json_create_only(r2_attempt, {"sentinel": False})
    with pytest.raises(FileExistsError, match="r2 single attempt"):
        r2._assert_fresh_canonical_artifacts(r2_result)
    assert all(not path.exists() for path in r1_paths)


@pytest.mark.parametrize("ccfr_pass", [True, False])
def test_r2_decision_is_the_frozen_r1_absolute_decision(
    ccfr_pass: bool,
) -> None:
    result = r2._assemble_r2_result(
        prerequisites={"sentinel": True},
        attempt_binding={"sentinel": True},
        ccfr={"all_pass": ccfr_pass},
        comparator={
            "objective_id": r1.COMPARATOR_ID,
            "execution_status": "ERROR",
            "all_pass": not ccfr_pass,
        },
        runtime={"sentinel": True},
    )

    assert result["all_pass"] is ccfr_pass
    assert result["decision"] == (
        "CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_PASS"
        if ccfr_pass
        else "CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_FAIL"
    )
    assert result["matched_v8_comparator_affects_decision"] is False
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert stable_fingerprint(unsigned) == fingerprint


def test_attempt_is_written_before_evaluate_and_no_retry_exists() -> None:
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    calls = [
        ast.unparse(node)
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
    ]
    source = EVALUATOR.read_text(encoding="utf-8")

    assert "_write_json_create_only(_CANONICAL_ATTEMPT, attempt)" in source
    assert source.index(
        "authority_token = _write_attempt_and_issue_authority(attempt)"
    ) < source.index(
        "result = evaluate(_authority_token=authority_token)"
    )
    assert "automatic_retry_allowed" in source
    assert "retry" not in " ".join(calls).lower()
    with pytest.raises(RuntimeError, match="only through canonical main"):
        r2.evaluate()
    with pytest.raises(RuntimeError, match="only through canonical main"):
        r2.evaluate(_authority_token=object())


def test_issued_authority_is_consumed_before_prerequisite_or_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopBeforeTraining(RuntimeError):
        pass

    monkeypatch.setattr(
        r2,
        "_CANONICAL_ATTEMPT",
        tmp_path / "attempt.json",
    )
    monkeypatch.setattr(r2, "_ACTIVE_AUTHORITY_TOKEN", None)
    token = r2._write_attempt_and_issue_authority(
        {"attempt_fingerprint": "a" * 64}
    )
    monkeypatch.setattr(
        r2,
        "_load_prerequisites",
        lambda: (_ for _ in ()).throw(StopBeforeTraining()),
    )

    with pytest.raises(StopBeforeTraining):
        r2.evaluate(_authority_token=token)
    with pytest.raises(RuntimeError, match="only through canonical main"):
        r2.evaluate(_authority_token=token)


def test_bound_authority_files_remain_byte_exact() -> None:
    failure = _load(FAILURE_RECEIPT)
    for binding in failure["frozen_upstream_bindings"].values():
        path = ROOT / binding["repo_path"]
        assert file_sha256(path) == binding["file_sha256"]


def test_complete_directly_binds_the_full_r2_hash_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_path = tmp_path / "attempt.json"
    result_path = tmp_path / "result.json"
    complete_path = tmp_path / "result.COMPLETE.sha256"
    failure_path = tmp_path / "failure.json"
    closure_path = tmp_path / "closure.json"
    pre_run_path = tmp_path / "pre-run.json"
    monkeypatch.setattr(r2, "_CANONICAL_ATTEMPT", attempt_path)
    monkeypatch.setattr(r2, "_CANONICAL_RESULT", result_path)
    monkeypatch.setattr(r2, "_CANONICAL_COMPLETE", complete_path)
    monkeypatch.setattr(r2, "_FAILURE_RECEIPT", failure_path)
    monkeypatch.setattr(r2, "_CORRECTION_CLOSURE", closure_path)
    monkeypatch.setattr(r2, "_PRE_RUN_RECEIPT", pre_run_path)

    attempt = {"sentinel": "attempt"}
    attempt["attempt_fingerprint"] = stable_fingerprint(attempt)
    result = {"sentinel": "result"}
    result["result_fingerprint"] = stable_fingerprint(result)
    r2._write_json_create_only(attempt_path, attempt)
    r2._write_json_create_only(result_path, result)

    receipts: list[tuple[Path, str]] = [
        (failure_path, "failure_receipt_fingerprint"),
        (closure_path, "closure_fingerprint"),
        (pre_run_path, "pre_run_fingerprint"),
    ]
    receipt_bindings: list[dict[str, str]] = []
    for path, field in receipts:
        value: dict[str, object] = {"sentinel": path.name}
        value[field] = stable_fingerprint(value)
        r2._write_json_create_only(path, value)
        receipt_bindings.append(
            {
                "file_sha256": file_sha256(path),
                field: value[field],
            }
        )
    prerequisites = {
        "r1_pre_attempt_failure": receipt_bindings[0],
        "r2_implementation_closure": receipt_bindings[1],
        "r2_pre_run_verification": receipt_bindings[2],
    }

    result_sha = r2._write_complete_create_only(
        result=result,
        attempt=attempt,
        prerequisites=prerequisites,
    )
    lines = complete_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    assert lines[0] == f"{result_sha}  result.json"
    assert lines[1] == f"attempt_sha256={file_sha256(attempt_path)}"
    assert lines[2:] == [
        f"attempt_fingerprint={attempt['attempt_fingerprint']}",
        "failure_receipt_sha256="
        f"{receipt_bindings[0]['file_sha256']}",
        "failure_receipt_fingerprint="
        f"{receipt_bindings[0]['failure_receipt_fingerprint']}",
        f"closure_sha256={receipt_bindings[1]['file_sha256']}",
        f"closure_fingerprint={receipt_bindings[1]['closure_fingerprint']}",
        f"pre_run_sha256={receipt_bindings[2]['file_sha256']}",
        f"pre_run_fingerprint={receipt_bindings[2]['pre_run_fingerprint']}",
        f"result_fingerprint={result['result_fingerprint']}",
    ]
    with pytest.raises(FileExistsError):
        r2._write_complete_create_only(
            result=result,
            attempt=attempt,
            prerequisites=prerequisites,
        )
