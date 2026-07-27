from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any

import pytest

from tools import audit_coverage_state_cmif_pmope_v18 as audit


_V17_ARTIFACTS = (
    "attempt.json",
    "checkpoints/identity_joint.checkpoint.json",
    "checkpoints/identity_joint.safetensors",
    "checkpoints/separable_endpoint.checkpoint.json",
    "checkpoints/separable_endpoint.safetensors",
    "checkpoints/support_oriented_response_joint.checkpoint.json",
    "checkpoints/support_oriented_response_joint.safetensors",
    "receipts/authorization.json",
    "receipts/bounded_result.json",
    "receipts/config.json",
    "receipts/dataset_free.json",
    "receipts/decision.json",
    "receipts/device_memory_preflight.json",
    "receipts/inputs.json",
    "receipts/preflight.json",
    "receipts/training.json",
    "receipts/zero_level.json",
)

_CONTROL_OBJECTIVES = (
    "support_oriented_response_joint",
    "identity_joint",
    "separable_endpoint",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _receipt(
    path: Path,
    payload: dict[str, Any],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, Any]:
    result = dict(payload)
    result[field] = audit._stable_fingerprint(result)
    _write_json(path, result)
    return result


def _sha(path: Path) -> str:
    return audit._file_sha256(path)


def _make_implementation(repo: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, relative in enumerate(audit.EXPECTED_IMPLEMENTATION_PATHS):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-source-{index}:{relative}\n".encode())
        result[relative] = _sha(path)
    return result


def _make_v17_closure(repo: Path) -> dict[str, Any]:
    root = repo / audit.V17_RUN_REPO_PATH
    (root / "checkpoints").mkdir(parents=True)
    (root / "receipts").mkdir()
    controls: list[dict[str, Any]] = []
    for objective in _CONTROL_OBJECTIVES:
        checkpoint = root / f"checkpoints/{objective}.safetensors"
        checkpoint.write_bytes(f"sealed-{objective}".encode())
        checkpoint_sha = _sha(checkpoint)
        module_fp = audit._stable_fingerprint(
            {"module": objective}
        )
        checkpoint_receipt = _receipt(
            root / f"checkpoints/{objective}.checkpoint.json",
            {
                "schema_version": "v17-checkpoint",
                "objective": objective,
                "objective_policy": f"policy-{objective}",
                "checkpoint_file_sha256": checkpoint_sha,
                "module_state_fingerprint": module_fp,
            },
        )
        controls.append(
            {
                "objective": objective,
                "objective_policy": f"policy-{objective}",
                "final_model_fingerprint": module_fp,
                "module_state_fingerprint": module_fp,
                "checkpoint_file_sha256": checkpoint_sha,
                "checkpoint_receipt_fingerprint": checkpoint_receipt[
                    "receipt_fingerprint"
                ],
                "zero_level_checkpoint_fingerprint": module_fp,
                "bounded_gate_passed": False,
            }
        )
    for relative in _V17_ARTIFACTS:
        path = root / relative
        if path.exists():
            continue
        _receipt(
            path,
            {
                "schema_version": "v17-fixture",
                "path": relative,
            },
        )
    artifacts = {
        relative: _sha(root / relative)
        for relative in _V17_ARTIFACTS
    }
    complete = _receipt(
        root / "COMPLETE.json",
        {
            "schema_version": "v17-complete",
            "status": "complete",
            "artifact_files": artifacts,
            "artifact_file_count": 17,
        },
        field="complete_fingerprint",
    )

    source_members = {
        f"source/member_{index:02d}.py": audit._stable_fingerprint(
            {"source": index}
        ).encode()
        for index in range(40)
    }
    archive_path = repo / audit.V17_SOURCE_ARCHIVE_REPO_PATH
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w:") as archive:
        for relative, content in source_members.items():
            info = tarfile.TarInfo(relative)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    member_hashes = {
        relative: audit.sha256(content).hexdigest()
        for relative, content in source_members.items()
    }
    archive_sha = _sha(archive_path)
    implementation_fp = audit._stable_fingerprint(member_hashes)
    manifest_path = repo / audit.V17_SOURCE_MANIFEST_REPO_PATH
    _write_json(
        manifest_path,
        {
            "schema_version": "cure-lite-cmif-source-closure-v1",
            "run_repo_path": audit.V17_RUN_REPO_PATH,
            "complete_file_sha256": _sha(root / "COMPLETE.json"),
            "complete_fingerprint": complete["complete_fingerprint"],
            "archive_repo_path": audit.V17_SOURCE_ARCHIVE_REPO_PATH,
            "archive_sha256": archive_sha,
            "source_file_count": 40,
            "implementation_fingerprint": implementation_fp,
        },
    )
    sealed = {
        "schema_version": "sealed-v17-fixture",
        "run_repo_path": audit.V17_RUN_REPO_PATH,
        "complete_fingerprint": complete["complete_fingerprint"],
        "complete_file_sha256": _sha(root / "COMPLETE.json"),
        "decision_fingerprint": audit._stable_fingerprint(
            {"decision": "v17"}
        ),
        "bounded_result_fingerprint": audit._stable_fingerprint(
            {"result": "v17"}
        ),
        "source_closure": {
            "manifest_file_sha256": _sha(manifest_path),
            "archive_file_sha256": archive_sha,
            "implementation_fingerprint": implementation_fp,
            "source_members": member_hashes,
        },
        "artifact_files": artifacts,
        "controls": controls,
        "checks": {"sealed": True},
        "all_pass": True,
        "historical_frozen_controls": True,
        "contemporaneous_controls": False,
        "control_outcomes_are_not_candidate_gates": True,
        "verification_mode": "read_only",
        "model_deserialization_performed": False,
        "evaluator_called": False,
        "training_performed": False,
        "D_R_cached_tensor_payload_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "runtime_splits": [],
    }
    return sealed


def _common_receipts(
    repo: Path,
    run: Path,
    *,
    dr_passed: bool,
) -> dict[str, dict[str, Any]]:
    implementation = _make_implementation(repo)
    sealed = _make_v17_closure(repo)
    config = _receipt(
        run / "receipts/config.json",
        {
            "schema_version": audit.RUN_SCHEMA,
            "run_id": audit.RUN_ID,
            "output_repo_path": audit.RUN_REPO_PATH,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "model": {
                "class": "CURELiteCenteredMixedInteractionLevelSet",
                "objective_suite": ["pmope_joint"],
                "candidate_objective": "pmope_joint",
                "candidate_objective_policy": "pmope-policy",
                "parameter_count": 64064,
                "field_threshold": 0.0,
                "threshold_search_performed": False,
            },
            "budget": {
                "seed": 42,
                "epochs": 10,
                "steps_per_epoch": 40,
                "updates_per_objective": 400,
                "objectives": 1,
            },
            "implementation": {
                "files": implementation,
                "implementation_fingerprint": (
                    audit._stable_fingerprint(implementation)
                ),
            },
            "evidence_scope": {
                "bounded_400_authorized": False,
                "formal_800_authorized": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
        },
    )
    attempt = _receipt(
        run / "attempt.json",
        {
            "schema_version": audit.ATTEMPT_SCHEMA,
            "run_id": audit.RUN_ID,
            "candidate_objective": "pmope_joint",
            "objectives": 1,
            "config_fingerprint": config["receipt_fingerprint"],
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    inputs = _receipt(
        run / "receipts/inputs.json",
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-inputs-v1"
            ),
            "runtime_splits": ["D_R"],
            "source_binding": {"split": "D_R"},
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    preflight = _receipt(
        run / "receipts/preflight.json",
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-preflight-v1"
            ),
            "training_authorized": True,
            "schedule": {"seed": 42},
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    dataset_free = _receipt(
        run / "receipts/dataset_free.json",
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-dataset-free-v1"
            ),
            "all_pass": True,
            "formal_800_authorized": False,
        },
    )
    sealed_wrapper = _receipt(
        run / "receipts/sealed_v17_controls.json",
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-"
                "sealed-v17-controls-v1"
            ),
            "sealed_v17": sealed,
            "sealed_v17_receipt_fingerprint": (
                audit._stable_fingerprint(sealed)
            ),
            "historical_controls_retrained": False,
            "historical_controls_reevaluated": False,
            "historical_control_outcomes_are_candidate_gates": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    dr_payload = {
        "schema_version": "pmope-dr-fixture",
        "checks": {"gate": dr_passed},
        "all_pass": dr_passed,
        "execution_seed": 42,
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    dr_gate = _receipt(
        run / "receipts/dr_gate.json",
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-"
                "real-D_R-gate-v1"
            ),
            "D_R_gate": dr_payload,
            "D_R_gate_evidence_fingerprint": (
                audit._stable_fingerprint(dr_payload)
            ),
            "all_pass": dr_passed,
            "gate_run_count": 1,
            "optimizer_steps": 0,
            "training_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    return {
        "config": config,
        "attempt": attempt,
        "inputs": inputs,
        "preflight": preflight,
        "dataset_free": dataset_free,
        "sealed": sealed_wrapper,
        "dr_gate": dr_gate,
    }


def _artifact_map(run: Path, paths: tuple[str, ...]) -> dict[str, str]:
    return {relative: _sha(run / relative) for relative in paths}


def _make_bounded_complete(repo: Path) -> Path:
    run = repo / audit.RUN_REPO_PATH
    (run / "checkpoints").mkdir(parents=True)
    (run / "receipts").mkdir()
    common = _common_receipts(repo, run, dr_passed=True)
    authorization = _receipt(
        run / "receipts/authorization.json",
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-authorization-v1"
            ),
            "authorization": {
                "objective_suite": ["pmope_joint"],
                "training_authorized": True,
            },
            "training_authorized": True,
            "formal_800_authorized": False,
        },
    )
    memory = _receipt(
        run / "receipts/device_memory_preflight.json",
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-"
                "device-memory-preflight-v1"
            ),
            "all_pass": True,
            "checks": {"cuda_available": True},
            "training_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    checkpoint = run / "checkpoints/pmope_joint.safetensors"
    checkpoint.write_bytes(b"tensor-only-fixture")
    checkpoint_receipt = _receipt(
        run / "checkpoints/pmope_joint.checkpoint.json",
        {
            "schema_version": audit.CHECKPOINT_SCHEMA,
            "objective": "pmope_joint",
            "objective_policy": "pmope-policy",
            "model_class": "CURELiteCenteredMixedInteractionLevelSet",
            "model_config": {
                "parameter_count": 64064,
                "fixed_margin_hex": None,
            },
            "repo_relative_path": (
                f"{audit.RUN_REPO_PATH}/"
                "checkpoints/pmope_joint.safetensors"
            ),
            "serialization": "safetensors",
            "tensor_only_state_dict": True,
            "weights_only_roundtrip_verified": True,
            "checkpoint_file_sha256": _sha(checkpoint),
            "module_state_fingerprint": audit._stable_fingerprint(
                {"module": "pmope"}
            ),
            "device_policy": "cpu_checkpoint",
        },
    )
    checkpoint_map = {
        "pmope_joint": checkpoint_receipt["receipt_fingerprint"]
    }
    training = _receipt(
        run / "receipts/training.json",
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-training-v1"
            ),
            "training": {
                "objective_suite": ["pmope_joint"],
                "objectives": [
                    {
                        "objective": "pmope_joint",
                        "seed": 42,
                        "epochs": 10,
                        "steps_per_epoch": 40,
                        "completed_updates": 400,
                        # Training and checkpoint fingerprints intentionally
                        # use different schemas in the real implementation.
                        "final_model_fingerprint": audit._stable_fingerprint(
                            {"training-model": "pmope"}
                        ),
                    }
                ],
            },
            "training_result_fingerprint": audit._stable_fingerprint(
                {"training": "pmope"}
            ),
            "checkpoint_receipt_fingerprints": checkpoint_map,
            "formal_training_performed": False,
            "bounded_training_performed": True,
            "candidate_count": 1,
            "candidate_objective": "pmope_joint",
            "historical_controls_retrained": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    zero = _receipt(
        run / "receipts/zero_level.json",
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-zero-level-v1"
            ),
            "candidate_objective": "pmope_joint",
            "candidate_bounded_gate_passed": True,
            "candidate_diagnostic": {
                "checkpoint_fingerprint": checkpoint_receipt[
                    "module_state_fingerprint"
                ],
            },
            "historical_controls_retrained": False,
            "historical_controls_reevaluated": False,
            "historical_control_outcomes_are_candidate_gates": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    result_fp = audit._stable_fingerprint({"bounded": "pmope"})
    bounded = _receipt(
        run / "receipts/bounded_result.json",
        {
            "schema_version": (
                "cure-lite-cmif-v18-pmope-bounded-400-result-v1"
            ),
            "result": {"candidate_objective": "pmope_joint"},
            "result_fingerprint": result_fp,
        },
    )
    decision = _receipt(
        run / "receipts/decision.json",
        {
            "schema_version": audit.DECISION_SCHEMA,
            "status": "PMOPE_V18_BOUNDED_400_GATE_PASS",
            "bounded_gate_passed": True,
            "candidate_gate_passed": True,
            "formal800_eligible": True,
            "result_fingerprint": result_fp,
            "checkpoint_receipt_fingerprints": checkpoint_map,
            "candidate_objective": "pmope_joint",
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
            "historical_controls_retrained": False,
            "historical_controls_reevaluated": False,
            "historical_control_outcomes_are_candidate_gates": False,
            "automatic_retry_allowed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    artifacts = _artifact_map(run, audit.EXPECTED_BOUNDED_ARTIFACT_PATHS)
    _receipt(
        run / "COMPLETE.json",
        {
            "schema_version": audit.RUN_SCHEMA,
            "status": "complete",
            "run_id": audit.RUN_ID,
            "decision": decision["status"],
            "bounded_gate_passed": True,
            "formal800_eligible": True,
            "D_R_gate_evidence_fingerprint": common["dr_gate"][
                "D_R_gate_evidence_fingerprint"
            ],
            "sealed_v17_evidence_fingerprint": common["sealed"][
                "sealed_v17_receipt_fingerprint"
            ],
            "config_fingerprint": common["config"]["receipt_fingerprint"],
            "input_receipt_fingerprint": common["inputs"][
                "receipt_fingerprint"
            ],
            "preflight_receipt_fingerprint": common["preflight"][
                "receipt_fingerprint"
            ],
            "dataset_free_receipt_fingerprint": common["dataset_free"][
                "receipt_fingerprint"
            ],
            "D_R_gate_receipt_fingerprint": common["dr_gate"][
                "receipt_fingerprint"
            ],
            "sealed_v17_receipt_fingerprint": common["sealed"][
                "receipt_fingerprint"
            ],
            "authorization_receipt_fingerprint": authorization[
                "receipt_fingerprint"
            ],
            "device_memory_preflight_receipt_fingerprint": memory[
                "receipt_fingerprint"
            ],
            "training_receipt_fingerprint": training[
                "receipt_fingerprint"
            ],
            "zero_level_receipt_fingerprint": zero["receipt_fingerprint"],
            "bounded_result_receipt_fingerprint": bounded[
                "receipt_fingerprint"
            ],
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifacts,
            "artifact_file_count": 15,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "performance_evaluation_performed": False,
        },
        field="complete_fingerprint",
    )
    return run


def _make_gate_stop(repo: Path) -> Path:
    run = repo / audit.RUN_REPO_PATH
    (run / "checkpoints").mkdir(parents=True)
    (run / "receipts").mkdir()
    common = _common_receipts(repo, run, dr_passed=False)
    decision = _receipt(
        run / "receipts/decision.json",
        {
            "schema_version": audit.DECISION_SCHEMA,
            "status": "PMOPE_V18_DR_GATE_FAIL",
            "bounded_gate_passed": False,
            "D_R_gate_passed": False,
            "D_R_gate_evidence_fingerprint": common["dr_gate"][
                "D_R_gate_evidence_fingerprint"
            ],
            "authorization_created": False,
            "bounded_training_performed": False,
            "checkpoint_count": 0,
            "formal800_eligible": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    artifacts = _artifact_map(
        run,
        audit.EXPECTED_GATE_STOP_ARTIFACT_PATHS,
    )
    _receipt(
        run / "COMPLETE.json",
        {
            "schema_version": audit.RUN_SCHEMA,
            "status": "complete",
            "run_id": audit.RUN_ID,
            "decision": "PMOPE_V18_DR_GATE_FAIL",
            "bounded_gate_passed": False,
            "D_R_gate_passed": False,
            "D_R_gate_evidence_fingerprint": common["dr_gate"][
                "D_R_gate_evidence_fingerprint"
            ],
            "config_fingerprint": common["config"]["receipt_fingerprint"],
            "input_receipt_fingerprint": common["inputs"][
                "receipt_fingerprint"
            ],
            "preflight_receipt_fingerprint": common["preflight"][
                "receipt_fingerprint"
            ],
            "dataset_free_receipt_fingerprint": common["dataset_free"][
                "receipt_fingerprint"
            ],
            "sealed_v17_receipt_fingerprint": common["sealed"][
                "receipt_fingerprint"
            ],
            "D_R_gate_receipt_fingerprint": common["dr_gate"][
                "receipt_fingerprint"
            ],
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifacts,
            "artifact_file_count": 8,
            "authorization_created": False,
            "bounded_training_performed": False,
            "checkpoint_count": 0,
            "formal800_eligible": False,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
        field="complete_fingerprint",
    )
    return run


def _make_failure(repo: Path) -> Path:
    run = repo / audit.RUN_REPO_PATH
    run.mkdir(parents=True)
    (run / ".incomplete").write_bytes(b"")
    attempt = _receipt(
        run / "attempt.json",
        {
            "schema_version": audit.ATTEMPT_SCHEMA,
            "run_id": audit.RUN_ID,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    _receipt(
        run / "FAILURE.json",
        {
            "schema_version": audit.FAILURE_SCHEMA,
            "status": "failed_incomplete_attempt",
            "exception_type": "RuntimeError",
            "message": "fixture failure",
            "attempt_fingerprint": attempt["receipt_fingerprint"],
            "artifact_files_before_failure": {
                "attempt.json": _sha(run / "attempt.json")
            },
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    )
    return run


def test_bounded_complete_exact_16_file_tree(tmp_path: Path) -> None:
    run = _make_bounded_complete(tmp_path)
    result = audit.audit_coverage_state_cmif_pmope_v18(
        run,
        repository_root=tmp_path,
    )
    assert result["terminal_state"] == "complete_bounded"
    assert result["artifact_file_count"] == 15
    assert result["tree_file_count"] == 16
    assert result["checkpoint_count"] == 1
    assert result["receipt_count"] == 12
    assert result["bounded_gate_passed"] is True
    assert result["checks"]["model_deserialization_performed"] is False
    assert result["checks"]["training_performed_by_auditor"] is False


def test_gate_stop_is_complete_without_training_or_checkpoint(
    tmp_path: Path,
) -> None:
    run = _make_gate_stop(tmp_path)
    result = audit.audit_coverage_state_cmif_pmope_v18(
        run,
        repository_root=tmp_path,
    )
    assert result["terminal_state"] == "complete_D_R_gate_stop"
    assert result["decision"] == "PMOPE_V18_DR_GATE_FAIL"
    assert result["artifact_file_count"] == 8
    assert result["tree_file_count"] == 9
    assert result["checkpoint_count"] == 0
    assert result["receipt_count"] == 7


def test_failure_requires_incomplete_and_forbids_complete(
    tmp_path: Path,
) -> None:
    run = _make_failure(tmp_path)
    result = audit.audit_coverage_state_cmif_pmope_v18(
        run,
        repository_root=tmp_path,
    )
    assert result["terminal_state"] == "failed_incomplete"
    assert result["decision"] is None
    assert result["checkpoint_count"] == 0
    (run / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="COMPLETE and FAILURE"):
        audit.audit_coverage_state_cmif_pmope_v18(
            run,
            repository_root=tmp_path,
        )


def test_checkpoint_byte_tampering_is_rejected(tmp_path: Path) -> None:
    run = _make_bounded_complete(tmp_path)
    (run / "checkpoints/pmope_joint.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact bytes changed"):
        audit.audit_coverage_state_cmif_pmope_v18(
            run,
            repository_root=tmp_path,
        )


def test_extra_terminal_file_is_rejected(tmp_path: Path) -> None:
    run = _make_gate_stop(tmp_path)
    (run / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(RuntimeError, match="gate-stop tree"):
        audit.audit_coverage_state_cmif_pmope_v18(
            run,
            repository_root=tmp_path,
        )


def test_implementation_source_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    run = _make_bounded_complete(tmp_path)
    relative = audit.EXPECTED_IMPLEMENTATION_PATHS[0]
    (tmp_path / relative).write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="implementation source changed"):
        audit.audit_coverage_state_cmif_pmope_v18(
            run,
            repository_root=tmp_path,
        )


def test_sealed_v17_archive_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    run = _make_gate_stop(tmp_path)
    archive = tmp_path / audit.V17_SOURCE_ARCHIVE_REPO_PATH
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="source closure bytes changed"):
        audit.audit_coverage_state_cmif_pmope_v18(
            run,
            repository_root=tmp_path,
        )


def test_cli_accepts_explicit_path_without_torch_dependency(
    tmp_path: Path,
) -> None:
    run = _make_gate_stop(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(audit.__file__)),
            str(run),
            "--repository-root",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["terminal_state"] == "complete_D_R_gate_stop"
    assert result["checks"]["model_deserialization_performed"] is False
    assert "torch" not in Path(audit.__file__).read_text(encoding="utf-8")
