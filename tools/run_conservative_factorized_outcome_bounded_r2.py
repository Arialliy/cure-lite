#!/usr/bin/env python3
"""Run the single versioned CC-SEA v8 verifier-correction bounded execution.

The r1 artifact is immutable and remains an execution/publication-contract
error.  This additive entrypoint changes only publication verification: it
accepts the executor's structured computational-gate records and verifies
their nested ``pass`` fields.  Model code, executor, inputs, schedules, loss,
optimizer budget, thresholds, device, and temperature policy remain bound to
the original v8 configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.config import LossConfig  # noqa: E402
from cure_lite.conservative_factorized_config import (  # noqa: E402
    ConservativeFactorizedDecoderConfig,
)
from cure_lite.experiment.conservative_factorized_result_verifier import (  # noqa: E402
    verify_conservative_factorized_core_result,
)
from tools import run_conservative_factorized_outcome_bounded as v1  # noqa: E402


METHOD_ID = "cc_sea_v8"
CORRECTION_ID = "cc_sea_v8_bounded_verifier_r2"
FROZEN_DEVICE = "cuda:0"

PROTOCOL_PREFIX = (
    "protocols/IRSTD-1K/"
    "coverage_conserving_subpixel_evidence_allocation_v8/"
)
CORRECTION_PREFIX = PROTOCOL_PREFIX + "verifier_correction_r2/"
PROPOSAL_REPO_PATH = CORRECTION_PREFIX + "proposal_receipt.json"
PROPOSAL_FILE_SHA256 = (
    "2f01541f7648e2fe68cd0974dad56f3e17808cbc69c3f8ab7dda17d46ba6ce53"
)
PROPOSAL_FINGERPRINT = (
    "18398de049d6f6428d0f8576b0c9d8452e8ff949120dbef5bb93108e4d8abd69"
)
CONFIG_REPO_PATH = CORRECTION_PREFIX + "config.json"
CONFIG_FILE_SHA256 = (
    "0dff997e26a501cd00cc0528b403f50542b33c7e9d03853d1af7d64b4d02ce9c"
)
CONFIG_FINGERPRINT = (
    "30c07ec68529506c71ca441595394a760cebdc49aac2f3f6c0d8d5644fdf170e"
)
CLOSURE_REPO_PATH = (
    CORRECTION_PREFIX + "implementation_closure_receipt.json"
)
AUTHORIZATION_REPO_PATH = CORRECTION_PREFIX + "run_authorization_receipt.json"

V1_CONFIG_REPO_PATH = PROTOCOL_PREFIX + "bounded_config.json"
V1_CONFIG_FILE_SHA256 = (
    "19ebde5b42643e65177084cb52d456e065e7ee9349852e1c68f4f6778a6c9b47"
)
V1_CONFIG_FINGERPRINT = (
    "baf120fdd7886877e70df3c2186035ab78df9c417aebe549250a142652b417ba"
)
V1_RUNNER_REPO_PATH = "tools/run_conservative_factorized_outcome_bounded.py"
V1_RUNNER_FILE_SHA256 = (
    "d24a7d2fed53b0c31f46d29dcb9a52cba9c59de29d2457dc8c244502878984cb"
)
V1_IMPLEMENTATION_FINGERPRINT = (
    "32453d343dc9157a3e8bdf206240767239b07a688d003e3d7826ea15bf03547a"
)
CORE_REPO_PATH = (
    "cure_lite/experiment/conservative_factorized_outcome_bounded.py"
)
CORE_FILE_SHA256 = (
    "63c9d2e447babb9dc57d8f78a0c8944733dc1e3d44dccb64301cbcd5cbd6b922"
)

R1_OUTPUT_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/cure_lite_cc_sea_v8_bounded_r1"
)
R1_COMPLETE_FILE_SHA256 = (
    "d8c3791d753bce73322636c77f765e7d3a5f03016b78d9242ca6aaa3ce4f6879"
)
R1_COMPLETE_FINGERPRINT = (
    "ae2f378c2f8a129861c8232313fc1e18c1b97e2c247ab7b8fc5cbf03daed6860"
)
R1_FAILURE_FILE_SHA256 = (
    "55f3c3c5f7b3d4a4dc35e2b4c99095bf0fb8c76f40b5ce973c169c75adfe0791"
)
R1_FAILURE_FINGERPRINT = (
    "a3a5cd2046b29fbb1a46d798ff2707a5eb22e1ca9c6a8ea14247e9fd529b06c0"
)

OUTPUT_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/cure_lite_cc_sea_v8_bounded_r2"
)
TEMPERATURE_WRAPPER_REPO_PATH = "tools/run_with_gpu_temperature_control.py"
TEMPERATURE_WRAPPER_FILE_SHA256 = (
    "026b751fbb59530721da1436af32f3bc924c9ed2ab3576df062a45bca7ec5e86"
)
PYTHON_EXECUTABLE = "/home/md0/ly/MSHNet/.venv/bin/python"

RUN_SCHEMA = "cure-lite-cc-sea-v8-bounded-correction-run-v1"
DECISION_SCHEMA = "cure-lite-cc-sea-v8-bounded-correction-decision-v1"
FAILURE_SCHEMA = "cure-lite-cc-sea-v8-bounded-correction-failure-v1"
RESULT_RECEIPT_SCHEMA = (
    "cure-lite-cc-sea-v8-bounded-correction-result-receipt-v1"
)
IMPLEMENTATION_SCHEMA = (
    "cure-lite-cc-sea-v8-verifier-correction-runtime-v1"
)
CLOSURE_SCHEMA = (
    "cure-lite-cc-sea-v8-verifier-correction-implementation-closure-v1"
)
AUTHORIZATION_SCHEMA = (
    "cure-lite-cc-sea-v8-verifier-correction-run-authorization-v1"
)

ROOT = Path(__file__).resolve().parents[1]
INCOMPLETE = ".incomplete"
PRE_RUN_RECEIPTS = {
    "authorization_binding.json",
    "config_binding.json",
    "implementation_binding.json",
    "implementation_closure_binding.json",
    "proposal_binding.json",
    "r1_attribution_binding.json",
    "run_claim.json",
    "v1_config_binding.json",
}
INPUT_RECEIPTS = {
    "anchor_population.json",
    "factual_schedule.json",
    "outcome_inputs.json",
    "outcome_schedule.json",
    "source_reconstruction.json",
}


@dataclass(frozen=True)
class PublishedCorrectionBounded:
    root: Path
    decision: str
    structural_execution_pass: bool
    bounded_model_code_gate_pass: bool
    complete_fingerprint: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _repo_file(repo_path: str, *, name: str) -> Path:
    return v1._repo_file(repo_path, name=name)


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    return v1._strict_json(path, name=name)


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    return v1._fingerprinted(payload, field=field)


def _verify_fingerprinted(
    payload: Mapping[str, Any],
    *,
    name: str,
    field: str = "receipt_fingerprint",
) -> None:
    v1._verify_fingerprinted(payload, name=name, field=field)


def _binding(
    *,
    schema: str,
    repo_path: str,
    path: Path,
    payload: Mapping[str, Any],
    fingerprint_field: str,
    payload_field: str,
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": schema,
            "method_id": METHOD_ID,
            "correction_id": CORRECTION_ID,
            "repo_path": repo_path,
            "file_sha256": file_sha256(path),
            "fingerprint_field": fingerprint_field,
            "bound_fingerprint": payload[fingerprint_field],
            payload_field: dict(payload),
        }
    )


def _load_exact(
    *,
    repo_path: str,
    sha256: str,
    fingerprint: str,
    fingerprint_field: str,
    name: str,
) -> tuple[dict[str, Any], Path]:
    path = _repo_file(repo_path, name=name)
    if file_sha256(path) != sha256:
        raise RuntimeError(f"{name} file SHA256 changed")
    payload = _strict_json(path, name=name)
    _verify_fingerprinted(
        payload,
        name=name,
        field=fingerprint_field,
    )
    if payload.get(fingerprint_field) != fingerprint:
        raise RuntimeError(f"{name} fingerprint changed")
    return payload, path


def _load_proposal() -> tuple[dict[str, Any], Path]:
    proposal, path = _load_exact(
        repo_path=PROPOSAL_REPO_PATH,
        sha256=PROPOSAL_FILE_SHA256,
        fingerprint=PROPOSAL_FINGERPRINT,
        fingerprint_field="proposal_fingerprint",
        name="CC-SEA v8 r2 correction proposal",
    )
    boundary = proposal.get("boundary")
    attribution = proposal.get("r1_failure_attribution")
    if (
        proposal.get("schema_version")
        != "cure-lite-cc-sea-v8-verifier-correction-proposal-v1"
        or proposal.get("method_id") != METHOD_ID
        or proposal.get("correction_id") != CORRECTION_ID
        or proposal.get("decision")
        != "CC_SEA_V8_R2_VERIFIER_CORRECTION_CODE_CREATION_AUTHORIZED"
        or not isinstance(boundary, Mapping)
        or boundary.get("D_R_payload_accessed") is not False
        or boundary.get("D_V_accessed") is not False
        or boundary.get("D_T_accessed") is not False
        or boundary.get("new_real_D_R_run_authorized") is not False
        or not isinstance(attribution, Mapping)
        or attribution.get("r1_is_not_a_model_nonpass") is not True
        or attribution.get("r1_run_claim_consumed") is not True
    ):
        raise RuntimeError("CC-SEA v8 r2 correction proposal changed")
    return proposal, path


def _load_config(path: Path) -> dict[str, Any]:
    canonical = v1._canonical_file(path, name="CC-SEA v8 r2 config")
    expected = _repo_file(CONFIG_REPO_PATH, name="CC-SEA v8 r2 config")
    if canonical != expected or file_sha256(canonical) != CONFIG_FILE_SHA256:
        raise RuntimeError("CC-SEA v8 r2 config file changed")
    config = _strict_json(canonical, name="CC-SEA v8 r2 config")
    _verify_fingerprinted(
        config,
        name="CC-SEA v8 r2 config",
        field="config_fingerprint",
    )
    execution = config.get("execution")
    scientific = config.get("scientific_contract")
    boundary = config.get("boundary")
    if (
        config.get("config_fingerprint") != CONFIG_FINGERPRINT
        or config.get("schema_version")
        != "cure-lite-cc-sea-v8-verifier-correction-config-v1"
        or config.get("method_id") != METHOD_ID
        or config.get("correction_id") != CORRECTION_ID
        or config.get("dataset") != "IRSTD-1K"
        or config.get("split") != "D_R"
        or not isinstance(execution, Mapping)
        or execution.get("device") != FROZEN_DEVICE
        or execution.get("output_repo_path") != OUTPUT_REPO_PATH
        or execution.get("exact_r2_run_count") != 1
        or execution.get("create_only") is not True
        or execution.get("resume_allowed") is not False
        or execution.get("automatic_retry_allowed") is not False
        or not isinstance(scientific, Mapping)
        or any(
            scientific.get(name) is not False
            for name in (
                "model_change_allowed",
                "core_executor_change_allowed",
                "loss_change_allowed",
                "budget_change_allowed",
                "threshold_change_allowed",
                "population_or_schedule_change_allowed",
            )
        )
        or not isinstance(boundary, Mapping)
        or boundary.get("D_R_payload_accessed") is not False
        or boundary.get("D_V_accessed") is not False
        or boundary.get("D_T_accessed") is not False
        or boundary.get("r2_execution_authorized") is not False
        or boundary.get("formal_800_authorized") is not False
    ):
        raise RuntimeError("CC-SEA v8 r2 config changed")
    return config


def _load_v1_config() -> tuple[dict[str, Any], Path]:
    config, path = _load_exact(
        repo_path=V1_CONFIG_REPO_PATH,
        sha256=V1_CONFIG_FILE_SHA256,
        fingerprint=V1_CONFIG_FINGERPRINT,
        fingerprint_field="config_fingerprint",
        name="CC-SEA v8 frozen v1 bounded config",
    )
    v1._validate_config_payload(config)
    return config, path


def _load_r1() -> v1.PublishedConservativeFactorizedOutcomeBounded:
    root = _repo_file(
        R1_OUTPUT_REPO_PATH + "/COMPLETE.json",
        name="CC-SEA v8 r1 COMPLETE",
    ).parent
    complete_path = root / "COMPLETE.json"
    failure_path = root / "receipts" / "failure.json"
    if (
        file_sha256(complete_path) != R1_COMPLETE_FILE_SHA256
        or file_sha256(failure_path) != R1_FAILURE_FILE_SHA256
    ):
        raise RuntimeError("CC-SEA v8 r1 frozen artifact changed")
    complete = _strict_json(complete_path, name="CC-SEA v8 r1 COMPLETE")
    failure = _strict_json(failure_path, name="CC-SEA v8 r1 failure")
    _verify_fingerprinted(
        complete,
        name="CC-SEA v8 r1 COMPLETE",
        field="complete_fingerprint",
    )
    _verify_fingerprinted(failure, name="CC-SEA v8 r1 failure")
    if (
        complete.get("complete_fingerprint") != R1_COMPLETE_FINGERPRINT
        or failure.get("receipt_fingerprint") != R1_FAILURE_FINGERPRINT
        or complete.get("decision")
        != "CC_SEA_V8_BOUNDED_EXECUTION_ERROR"
        or complete.get("real_D_R_run_claim_consumed") is not True
        or failure.get("message")
        != "CC-SEA v8 full bounded evidence changed"
    ):
        raise RuntimeError("CC-SEA v8 r1 attribution changed")
    return v1.load_conservative_factorized_outcome_bounded_artifact(root)


def _implementation_binding() -> dict[str, object]:
    inherited = v1._implementation_binding()
    inherited_files = inherited.get("all_runtime_files")
    if (
        not isinstance(inherited_files, Mapping)
        or stable_fingerprint(inherited) != V1_IMPLEMENTATION_FINGERPRINT
    ):
        raise RuntimeError("CC-SEA v8 r1 runtime binding changed")
    if (
        file_sha256(_repo_file(V1_RUNNER_REPO_PATH, name="v1 runner"))
        != V1_RUNNER_FILE_SHA256
        or file_sha256(_repo_file(CORE_REPO_PATH, name="v8 core"))
        != CORE_FILE_SHA256
    ):
        raise RuntimeError("CC-SEA v8 frozen executor or v1 runner changed")
    additive_paths = (
        "cure_lite/experiment/conservative_factorized_result_verifier.py",
        "tools/run_conservative_factorized_outcome_bounded_r2.py",
    )
    additive_files = {
        repo_path: file_sha256(
            _repo_file(repo_path, name="CC-SEA v8 r2 runtime file")
        )
        for repo_path in additive_paths
    }
    if set(additive_files) & set(inherited_files):
        raise RuntimeError("CC-SEA v8 r2 files are not additive")
    all_files = {**dict(inherited_files), **additive_files}
    return {
        "schema_version": IMPLEMENTATION_SCHEMA,
        "method_id": METHOD_ID,
        "correction_id": CORRECTION_ID,
        "v1_implementation_fingerprint": stable_fingerprint(inherited),
        "v1_runtime_files": dict(sorted(inherited_files.items())),
        "additive_runtime_files": dict(sorted(additive_files.items())),
        "all_runtime_files": dict(sorted(all_files.items())),
        "model_or_core_file_changed": False,
    }


def _verify_implementation_files(binding: Mapping[str, Any]) -> None:
    files = binding.get("all_runtime_files")
    if not isinstance(files, Mapping):
        raise RuntimeError("CC-SEA v8 r2 runtime inventory is missing")
    for repo_path, digest in files.items():
        if (
            not isinstance(repo_path, str)
            or not isinstance(digest, str)
            or file_sha256(
                _repo_file(repo_path, name="CC-SEA v8 r2 bound file")
            )
            != digest
        ):
            raise RuntimeError(
                f"CC-SEA v8 r2 runtime file changed: {repo_path}"
            )


def _load_closure(
    implementation: Mapping[str, object],
) -> tuple[dict[str, Any], Path]:
    path = _repo_file(CLOSURE_REPO_PATH, name="CC-SEA v8 r2 closure")
    closure = _strict_json(path, name="CC-SEA v8 r2 closure")
    _verify_fingerprinted(closure, name="CC-SEA v8 r2 closure")
    boundary = closure.get("boundary")
    tests = closure.get("test_evidence")
    if (
        closure.get("schema_version") != CLOSURE_SCHEMA
        or closure.get("method_id") != METHOD_ID
        or closure.get("correction_id") != CORRECTION_ID
        or closure.get("decision")
        != "CC_SEA_V8_R2_VERIFIER_CORRECTION_IMPLEMENTATION_PASS"
        or closure.get("runtime_implementation_binding")
        != _fingerprinted(implementation)
        or not isinstance(tests, Mapping)
        or tests.get("all_required_tests_passed") is not True
        or tests.get("direct_core_to_real_verifier_passed") is not True
        or tests.get("r1_strict_loader_regression_passed") is not True
        or not isinstance(boundary, Mapping)
        or boundary.get("D_R_payload_accessed") is not False
        or boundary.get("D_V_accessed") is not False
        or boundary.get("D_T_accessed") is not False
        or boundary.get("r2_execution_authorized") is not False
        or boundary.get("formal_800_authorized") is not False
    ):
        raise RuntimeError("CC-SEA v8 r2 implementation closure changed")
    return closure, path


def _load_authorization(
    config: Mapping[str, Any],
    closure: Mapping[str, Any],
    closure_path: Path,
    implementation: Mapping[str, object],
) -> tuple[dict[str, Any], Path]:
    path = _repo_file(
        AUTHORIZATION_REPO_PATH,
        name="CC-SEA v8 r2 authorization",
    )
    authorization = _strict_json(path, name="CC-SEA v8 r2 authorization")
    _verify_fingerprinted(
        authorization,
        name="CC-SEA v8 r2 authorization",
    )
    permission = authorization.get("authorization")
    boundary = authorization.get("boundary")
    execution = authorization.get("execution_control_binding")
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or authorization.get("method_id") != METHOD_ID
        or authorization.get("correction_id") != CORRECTION_ID
        or authorization.get("decision")
        != "CC_SEA_V8_R2_ONE_CORRECTIVE_D_R_RUN_AUTHORIZED"
        or authorization.get("config_fingerprint") != CONFIG_FINGERPRINT
        or authorization.get("implementation_fingerprint")
        != stable_fingerprint(implementation)
        or authorization.get("closure_fingerprint")
        != closure.get("receipt_fingerprint")
        or authorization.get("closure_file_sha256")
        != file_sha256(closure_path)
        or not isinstance(permission, Mapping)
        or permission.get("exact_r2_run_count") != 1
        or permission.get("output_repo_path") != OUTPUT_REPO_PATH
        or permission.get("device") != FROZEN_DEVICE
        or permission.get("create_only") is not True
        or permission.get("resume_allowed") is not False
        or permission.get("automatic_retry_allowed") is not False
        or permission.get("D_V_access_allowed") is not False
        or permission.get("D_T_access_allowed") is not False
        or permission.get("formal_800_allowed") is not False
        or not isinstance(boundary, Mapping)
        or boundary.get("r1_remains_consumed_and_immutable") is not True
        or boundary.get("r2_is_not_an_additional_seed") is not True
        or boundary.get("r2_pass_or_nonpass_must_be_frozen") is not True
        or not isinstance(execution, Mapping)
        or execution.get("gpu_index") != 0
        or execution.get("pause_temperature_celsius") != 82
        or execution.get("resume_temperature_celsius") != 75
        or execution.get("wrapper_repo_path")
        != TEMPERATURE_WRAPPER_REPO_PATH
        or execution.get("wrapper_file_sha256")
        != TEMPERATURE_WRAPPER_FILE_SHA256
    ):
        raise RuntimeError("CC-SEA v8 r2 authorization changed")
    if config.get("config_fingerprint") != CONFIG_FINGERPRINT:
        raise RuntimeError("CC-SEA v8 r2 authorization/config mismatch")
    return authorization, path


def _validate_device(value: object) -> str:
    if value != FROZEN_DEVICE:
        raise ValueError("CC-SEA v8 r2 fixes --device at cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError(
            "CC-SEA v8 r2 requires the frozen GPU-0 temperature wrapper"
        )
    return FROZEN_DEVICE


def _validate_output_target(path: Path) -> Path:
    candidate = Path(os.path.abspath(path.expanduser()))
    expected = Path(os.path.abspath(ROOT / OUTPUT_REPO_PATH))
    if candidate != expected:
        raise ValueError(f"CC-SEA v8 r2 permits only {expected}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"CC-SEA v8 r2 output exists: {candidate}")
    for parent in (candidate.parent, *candidate.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("CC-SEA v8 r2 output may not traverse a symlink")
    return candidate


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {INCOMPLETE, "COMPLETE.json"}
    }


def _write_new(path: Path, payload: Mapping[str, object]) -> None:
    v1._write_new_json(path, payload)


def _pre_run_receipts(
    *,
    proposal: Mapping[str, Any],
    proposal_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    v1_config: Mapping[str, Any],
    v1_config_path: Path,
    implementation: Mapping[str, object],
    closure: Mapping[str, Any],
    closure_path: Path,
    authorization: Mapping[str, Any],
    authorization_path: Path,
) -> dict[str, dict[str, object]]:
    return {
        "proposal_binding.json": _binding(
            schema="cure-lite-cc-sea-v8-r2-proposal-binding-v1",
            repo_path=PROPOSAL_REPO_PATH,
            path=proposal_path,
            payload=proposal,
            fingerprint_field="proposal_fingerprint",
            payload_field="proposal",
        ),
        "config_binding.json": _binding(
            schema="cure-lite-cc-sea-v8-r2-config-binding-v1",
            repo_path=CONFIG_REPO_PATH,
            path=config_path,
            payload=config,
            fingerprint_field="config_fingerprint",
            payload_field="config",
        ),
        "v1_config_binding.json": _binding(
            schema="cure-lite-cc-sea-v8-r2-v1-config-binding-v1",
            repo_path=V1_CONFIG_REPO_PATH,
            path=v1_config_path,
            payload=v1_config,
            fingerprint_field="config_fingerprint",
            payload_field="v1_config",
        ),
        "implementation_binding.json": _fingerprinted(implementation),
        "implementation_closure_binding.json": _binding(
            schema="cure-lite-cc-sea-v8-r2-closure-binding-v1",
            repo_path=CLOSURE_REPO_PATH,
            path=closure_path,
            payload=closure,
            fingerprint_field="receipt_fingerprint",
            payload_field="closure",
        ),
        "authorization_binding.json": _binding(
            schema="cure-lite-cc-sea-v8-r2-authorization-binding-v1",
            repo_path=AUTHORIZATION_REPO_PATH,
            path=authorization_path,
            payload=authorization,
            fingerprint_field="receipt_fingerprint",
            payload_field="authorization",
        ),
        "r1_attribution_binding.json": _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cc-sea-v8-r2-r1-attribution-binding-v1"
                ),
                "method_id": METHOD_ID,
                "correction_id": CORRECTION_ID,
                "r1_output_repo_path": R1_OUTPUT_REPO_PATH,
                "r1_complete_file_sha256": R1_COMPLETE_FILE_SHA256,
                "r1_complete_fingerprint": R1_COMPLETE_FINGERPRINT,
                "r1_failure_file_sha256": R1_FAILURE_FILE_SHA256,
                "r1_failure_fingerprint": R1_FAILURE_FINGERPRINT,
                "r1_status": "EXECUTOR_RESULT_TO_PUBLICATION_CONTRACT_ERROR",
                "r1_is_not_a_model_nonpass": True,
                "r1_run_claim_consumed": True,
            }
        ),
        "run_claim.json": _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cc-sea-v8-r2-single-run-claim-v1"
                ),
                "method_id": METHOD_ID,
                "correction_id": CORRECTION_ID,
                "split": "D_R",
                "device": FROZEN_DEVICE,
                "claim_consumed_before_D_R_loader": True,
                "exact_r2_run_count_claimed": 1,
                "r2_is_not_an_additional_seed": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        ),
    }


def _decision(
    result: Mapping[str, object] | None,
    *,
    failure: Mapping[str, object] | None,
    evidence_fingerprint: str,
) -> dict[str, object]:
    if result is None:
        status = "CC_SEA_V8_R2_BOUNDED_EXECUTION_ERROR"
        structural = False
        model_pass = False
        kind = "failure"
    else:
        structural = result.get("structural_execution_pass") is True
        model_pass = (
            result.get("computational_model_code_gate_pass") is True
        )
        status = (
            "CC_SEA_V8_R2_BOUNDED_MODEL_CODE_GATE_PASS"
            if model_pass
            else (
                "CC_SEA_V8_R2_BOUNDED_MODEL_CODE_GATE_FAIL"
                if structural
                else "CC_SEA_V8_R2_STRUCTURAL_EXECUTION_FAIL"
            )
        )
        kind = "result"
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "method_id": METHOD_ID,
            "correction_id": CORRECTION_ID,
            "status": status,
            "structural_execution_pass": structural,
            "bounded_model_code_gate_pass": model_pass,
            "evidence_kind": kind,
            "evidence_receipt_fingerprint": evidence_fingerprint,
            "failure": None if failure is None else dict(failure),
            "not_detection_performance_evidence": True,
            "directly_authorizes_formal_800": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "next_action": (
                "freeze_and_review_r2_bounded_model_code_evidence"
                if model_pass
                else "freeze_r2_result_and_stop_without_another_run"
            ),
        }
    )


def load_correction_bounded_artifact(
    output_dir: str | Path,
    *,
    _allow_incomplete: bool = False,
) -> PublishedCorrectionBounded:
    root = Path(output_dir).expanduser().resolve(strict=True)
    if root != Path(os.path.abspath(output_dir)) or not root.is_dir():
        raise ValueError("CC-SEA v8 r2 artifact root is not canonical")
    incomplete = (root / INCOMPLETE).exists()
    if incomplete and not _allow_incomplete:
        raise RuntimeError("CC-SEA v8 r2 publication is incomplete")
    expected_top = {"receipts", "COMPLETE.json"}
    if _allow_incomplete:
        expected_top.add(INCOMPLETE)
    if {item.name for item in root.iterdir()} != expected_top:
        raise RuntimeError("CC-SEA v8 r2 top-level inventory changed")
    receipts_root = root / "receipts"
    names = {item.name for item in receipts_root.iterdir()}
    full_result = PRE_RUN_RECEIPTS | INPUT_RECEIPTS | {
        "decision.json",
        "result.json",
    }
    full_failure = PRE_RUN_RECEIPTS | INPUT_RECEIPTS | {
        "decision.json",
        "failure.json",
    }
    if names not in (full_result, full_failure):
        raise RuntimeError("CC-SEA v8 r2 receipt inventory changed")
    payloads = {
        name[:-5]: _strict_json(
            receipts_root / name,
            name=f"CC-SEA v8 r2 {name}",
        )
        for name in names
    }
    for name, payload in payloads.items():
        _verify_fingerprinted(payload, name=f"CC-SEA v8 r2 {name}")
    complete = _strict_json(root / "COMPLETE.json", name="r2 COMPLETE")
    _verify_fingerprinted(
        complete,
        name="CC-SEA v8 r2 COMPLETE",
        field="complete_fingerprint",
    )
    if (
        complete.get("schema_version") != RUN_SCHEMA
        or complete.get("method_id") != METHOD_ID
        or complete.get("correction_id") != CORRECTION_ID
        or complete.get("execution_status") != "complete"
        or complete.get("artifact_files") != _artifact_hashes(root)
        or complete.get("artifact_file_count") != len(_artifact_hashes(root))
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("performance_evaluation_performed") is not False
        or complete.get("formal_800_training_performed") is not False
        or complete.get("r2_run_count") != 1
        or complete.get("r2_run_claim_consumed") is not True
        or complete.get("r1_remains_immutable") is not True
    ):
        raise RuntimeError("CC-SEA v8 r2 COMPLETE changed")

    proposal, proposal_path = _load_proposal()
    config_path = _repo_file(CONFIG_REPO_PATH, name="r2 config")
    config = _load_config(config_path)
    v1_config, _ = _load_v1_config()
    _load_r1()
    implementation = _implementation_binding()
    _verify_implementation_files(implementation)
    closure, closure_path = _load_closure(implementation)
    authorization, _ = _load_authorization(
        config,
        closure,
        closure_path,
        implementation,
    )
    if (
        payloads["proposal_binding"].get("proposal") != proposal
        or payloads["config_binding"].get("config") != config
        or payloads["v1_config_binding"].get("v1_config") != v1_config
        or payloads["implementation_binding"] != _fingerprinted(implementation)
        or payloads["implementation_closure_binding"].get("closure")
        != closure
        or payloads["authorization_binding"].get("authorization")
        != authorization
        or file_sha256(proposal_path) != PROPOSAL_FILE_SHA256
    ):
        raise RuntimeError("CC-SEA v8 r2 embedded binding changed")

    if "result" in payloads:
        result_receipt = payloads["result"]
        core_result = result_receipt.get("core_result")
        if (
            result_receipt.get("schema_version") != RESULT_RECEIPT_SCHEMA
            or not isinstance(core_result, Mapping)
        ):
            raise RuntimeError("CC-SEA v8 r2 result receipt changed")
        verify_conservative_factorized_core_result(core_result)
        structural = core_result.get("structural_execution_pass") is True
        model_pass = (
            core_result.get("computational_model_code_gate_pass") is True
        )
        evidence_kind = "result"
        evidence_fingerprint = result_receipt["receipt_fingerprint"]
    else:
        failure = payloads["failure"]
        if (
            failure.get("schema_version") != FAILURE_SCHEMA
            or failure.get("r2_run_claim_consumed") is not True
            or failure.get("D_V_accessed") is not False
            or failure.get("D_T_accessed") is not False
        ):
            raise RuntimeError("CC-SEA v8 r2 failure receipt changed")
        structural = False
        model_pass = False
        evidence_kind = "failure"
        evidence_fingerprint = failure["receipt_fingerprint"]
    expected_status = (
        "CC_SEA_V8_R2_BOUNDED_MODEL_CODE_GATE_PASS"
        if model_pass
        else (
            "CC_SEA_V8_R2_BOUNDED_MODEL_CODE_GATE_FAIL"
            if structural
            else (
                "CC_SEA_V8_R2_BOUNDED_EXECUTION_ERROR"
                if evidence_kind == "failure"
                else "CC_SEA_V8_R2_STRUCTURAL_EXECUTION_FAIL"
            )
        )
    )
    decision = payloads["decision"]
    if (
        decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("status") != expected_status
        or decision.get("structural_execution_pass") is not structural
        or decision.get("bounded_model_code_gate_pass") is not model_pass
        or decision.get("evidence_kind") != evidence_kind
        or decision.get("evidence_receipt_fingerprint")
        != evidence_fingerprint
        or complete.get("decision") != expected_status
        or complete.get("decision_fingerprint")
        != decision.get("receipt_fingerprint")
    ):
        raise RuntimeError("CC-SEA v8 r2 decision binding changed")
    return PublishedCorrectionBounded(
        root=root,
        decision=expected_status,
        structural_execution_pass=structural,
        bounded_model_code_gate_pass=model_pass,
        complete_fingerprint=str(complete["complete_fingerprint"]),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = v1._canonical_file(args.config, name="CC-SEA v8 r2 config")
    config = _load_config(config_path)
    device = _validate_device(args.device)
    proposal, proposal_path = _load_proposal()
    v1_config, v1_config_path = _load_v1_config()
    _load_r1()
    implementation = _implementation_binding()
    _verify_implementation_files(implementation)
    closure, closure_path = _load_closure(implementation)
    authorization, authorization_path = _load_authorization(
        config,
        closure,
        closure_path,
        implementation,
    )
    output = _validate_output_target(args.output)

    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / INCOMPLETE
    incomplete.open("xb").close()
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    for name, payload in _pre_run_receipts(
        proposal=proposal,
        proposal_path=proposal_path,
        config=config,
        config_path=config_path,
        v1_config=v1_config,
        v1_config_path=v1_config_path,
        implementation=implementation,
        closure=closure,
        closure_path=closure_path,
        authorization=authorization,
        authorization_path=authorization_path,
    ).items():
        _write_new(receipts / name, payload)

    real_inputs: v1._FrozenRealInputs | None = None
    core_result: dict[str, object] | None = None
    execution_error: Exception | None = None
    failure_phase = "D_R_RECONSTRUCTION"
    try:
        real_inputs = v1._load_frozen_real_inputs(v1_config)
        input_receipts = {
            "source_reconstruction.json": _fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-cc-sea-v8-r2-source-reconstruction-v1"
                    ),
                    "method_id": METHOD_ID,
                    "correction_id": CORRECTION_ID,
                    "split": "D_R",
                    "source_config_repo_path": (
                        real_inputs.source_config_path.relative_to(ROOT).as_posix()
                    ),
                    "source_config_file_sha256": file_sha256(
                        real_inputs.source_config_path
                    ),
                    "source_config_fingerprint": real_inputs.source_config[
                        "config_fingerprint"
                    ],
                    "pair_catalog_fingerprint": (
                        real_inputs.pair_catalog_fingerprint
                    ),
                    "prepared_catalog_fingerprint": (
                        real_inputs.prepared_catalog_fingerprint
                    ),
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                }
            ),
            "anchor_population.json": _fingerprinted(
                real_inputs.population.canonical_receipt()
            ),
            "factual_schedule.json": _fingerprinted(
                real_inputs.factual_schedule.canonical_receipt()
            ),
            "outcome_inputs.json": _fingerprinted(
                real_inputs.materializer.canonical_receipt()
            ),
            "outcome_schedule.json": _fingerprinted(
                real_inputs.outcome_schedule.canonical_receipt()
            ),
        }
        for name, payload in input_receipts.items():
            _write_new(receipts / name, payload)

        failure_phase = "BOUNDED_EXECUTION"
        core_result = v1.execute_conservative_factorized_outcome_bounded(
            real_inputs.population,
            real_inputs.factual_schedule,
            real_inputs.outcome_schedule,
            real_inputs.materializer,
            ConservativeFactorizedDecoderConfig(
                **v1_config["optimization"]["decoder"]
            ),
            LossConfig(**v1_config["optimization"]["loss"]),
            v1._optimization_budget(v1_config),
            device=device,
            evaluation_chunk_size=v1_config["budget"][
                "evaluation_chunk_size"
            ],
        )
        verify_conservative_factorized_core_result(core_result)
    except Exception as error:
        execution_error = error

    post_attempt_error: Exception | None = None
    try:
        if real_inputs is not None:
            real_inputs.bundle.verify_unchanged()
            if any(
                file_sha256(Path(path)) != digest
                for path, digest in real_inputs.immutable.items()
            ):
                raise RuntimeError("a frozen r2 D_R input changed")
        if (
            _load_config(config_path) != config
            or _load_proposal()[0] != proposal
            or _load_v1_config()[0] != v1_config
            or _implementation_binding() != implementation
        ):
            raise RuntimeError("CC-SEA v8 r2 static inputs changed")
        current_closure, current_closure_path = _load_closure(
            implementation
        )
        current_authorization, current_authorization_path = (
            _load_authorization(
                config,
                closure,
                closure_path,
                implementation,
            )
        )
        _load_r1()
        if (
            current_closure != closure
            or file_sha256(current_closure_path)
            != file_sha256(closure_path)
            or current_authorization != authorization
            or file_sha256(current_authorization_path)
            != file_sha256(authorization_path)
        ):
            raise RuntimeError("CC-SEA v8 r2 authorization changed")
    except Exception as error:
        post_attempt_error = error
        if execution_error is None:
            execution_error = error
            failure_phase = "POST_EXECUTION_IMMUTABILITY"

    if execution_error is None:
        if core_result is None:
            raise RuntimeError("CC-SEA v8 r2 returned no core result")
        evidence = _fingerprinted(
            {
                "schema_version": RESULT_RECEIPT_SCHEMA,
                "method_id": METHOD_ID,
                "correction_id": CORRECTION_ID,
                "core_result": core_result,
            }
        )
        _write_new(receipts / "result.json", evidence)
        failure = None
    else:
        core_result = None
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "method_id": METHOD_ID,
            "correction_id": CORRECTION_ID,
            "phase": failure_phase,
            "exception_type": type(execution_error).__name__,
            "message": str(execution_error),
            "post_attempt_verification_passed": (
                post_attempt_error is None
            ),
            "post_attempt_exception_type": (
                None
                if post_attempt_error is None
                else type(post_attempt_error).__name__
            ),
            "post_attempt_exception_message": (
                None
                if post_attempt_error is None
                else str(post_attempt_error)
            ),
            "r2_run_claim_consumed": True,
            "structural_execution_pass": False,
            "bounded_model_code_gate_pass": False,
            "model_or_core_changed": False,
            "budget_or_threshold_changed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
        evidence = _fingerprinted(failure)
        _write_new(receipts / "failure.json", evidence)

    decision = _decision(
        core_result,
        failure=failure,
        evidence_fingerprint=str(evidence["receipt_fingerprint"]),
    )
    _write_new(receipts / "decision.json", decision)
    artifacts = _artifact_hashes(output)
    complete = _fingerprinted(
        {
            "schema_version": RUN_SCHEMA,
            "method_id": METHOD_ID,
            "correction_id": CORRECTION_ID,
            "execution_status": "complete",
            "decision": decision["status"],
            "structural_execution_pass": decision[
                "structural_execution_pass"
            ],
            "bounded_model_code_gate_pass": decision[
                "bounded_model_code_gate_pass"
            ],
            "device": device,
            "split": "D_R",
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
            "formal_800_training_performed": False,
            "resume_used": False,
            "automatic_retry_performed": False,
            "r2_run_count": 1,
            "r2_run_claim_consumed": True,
            "r2_is_not_an_additional_seed": True,
            "r1_remains_immutable": True,
            "r1_complete_fingerprint": R1_COMPLETE_FINGERPRINT,
            "post_attempt_verification_passed": (
                post_attempt_error is None
            ),
            "input_receipts_present": real_inputs is not None,
            "config_fingerprint": CONFIG_FINGERPRINT,
            "proposal_fingerprint": PROPOSAL_FINGERPRINT,
            "implementation_fingerprint": stable_fingerprint(
                implementation
            ),
            "closure_fingerprint": closure["receipt_fingerprint"],
            "authorization_fingerprint": authorization[
                "receipt_fingerprint"
            ],
            "evidence_kind": decision["evidence_kind"],
            "evidence_receipt_fingerprint": evidence[
                "receipt_fingerprint"
            ],
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifacts,
            "artifact_file_count": len(artifacts),
        },
        field="complete_fingerprint",
    )
    _write_new(output / "COMPLETE.json", complete)
    published = load_correction_bounded_artifact(
        output,
        _allow_incomplete=True,
    )
    incomplete.unlink()
    return {
        "output": str(output),
        "decision": published.decision,
        "structural_execution_pass": (
            published.structural_execution_pass
        ),
        "bounded_model_code_gate_pass": (
            published.bounded_model_code_gate_pass
        ),
        "complete_fingerprint": published.complete_fingerprint,
        "r2_run_claim_consumed": True,
        "r1_remains_immutable": True,
        "not_detection_performance_evidence": True,
        "directly_authorizes_formal_800": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if result["bounded_model_code_gate_pass"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
