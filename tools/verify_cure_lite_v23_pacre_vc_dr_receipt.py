#!/usr/bin/env python3
"""Verify the terminal PACRE-VC D_R artifact graph without reopening tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite_v23.authorization import (
    PACRE_VC_DR_OUTPUT_REPO_PATH,
    PACRE_VC_DR_RUN_ID,
    dr_output_path,
    protocol_root,
    verify_dr_pre_run_authorization,
)
from cure_lite_v23.dr_gate import (
    pacre_vc_dr_receipt_from_payload,
    recompute_pacre_vc_dr_checks,
)
from cure_lite_v23.protocol import (
    read_strict_json,
    verify_fingerprinted,
)

PREREQUISITE_FILES = {
    "v22_failure_inheritance": "v22_failure_inheritance_receipt.json",
    "runtime_cpu": "runtime_environment_cpu_lock.json",
    "runtime_selected_device": "runtime_environment_lock.json",
    "forward_parity": "forward_parity_receipt.json",
    "scalar_counterexample": "scalar_counterexample_receipt.json",
    "cpu_stress": "formal_shape_cpu_stress_receipt.json",
    "selected_device_stress": (
        "formal_shape_selected_device_stress_receipt.json"
    ),
    "dataset_free": "dataset_free_receipt.json",
    "implementation_closure": "implementation_closure.json",
    "runner_verification": "runner_verification_receipt.json",
}
EXPECTED_DIRECTORIES = frozenset({"receipts"})
EXPECTED_ARTIFACT_FILES = frozenset(
    {
        "attempt.json",
        "receipts/inputs.json",
        "receipts/preflight.json",
        "receipts/dr_gate.json",
        "receipts/decision.json",
    }
)


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    if frozenset(payload) != expected:
        raise ValueError(f"{name} fields differ from the fixed schema")


def _verify_output_population(output: Path) -> dict[str, str]:
    directories: set[str] = set()
    files: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        relative = str(path.relative_to(output))
        if path.is_symlink():
            raise RuntimeError("PACRE-VC terminal contains a symlink")
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files[relative] = file_sha256(path)
        else:
            raise RuntimeError("PACRE-VC terminal contains a special file")
    expected_files = EXPECTED_ARTIFACT_FILES | {"COMPLETE.json"}
    if directories != EXPECTED_DIRECTORIES or set(files) != expected_files:
        raise RuntimeError(
            "PACRE-VC terminal artifact population differs"
        )
    return files


def _read_prerequisites() -> dict[str, dict[str, object]]:
    root = protocol_root()
    return {
        name: read_strict_json(root / filename)
        for name, filename in PREREQUISITE_FILES.items()
    }


def _verify_persisted_input_bindings(
    inputs: Mapping[str, object],
    *,
    authorization: Mapping[str, object],
    authorization_fingerprint: str,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    _require_exact_fields(
        inputs,
        frozenset(
            {
                "schema_version",
                "run_id",
                "authorization_fingerprint",
                "real_inputs",
                "real_inputs_fingerprint",
                "population",
                "population_fingerprint",
                "cache_fingerprint",
                "source_files",
                "construction_invocations",
                "D_R_accessed",
                "D_V_accessed",
                "D_T_accessed",
                "split_manifest_metadata_read",
                "D_V_tensor_payload_accessed",
                "D_T_tensor_payload_accessed",
                "training_performed",
                "receipt_fingerprint",
            }
        ),
        name="D_R inputs receipt",
    )
    verify_fingerprinted(inputs)
    real_payload = inputs.get("real_inputs")
    population_payload = inputs.get("population")
    if not isinstance(real_payload, Mapping) or not isinstance(
        population_payload,
        Mapping,
    ):
        raise TypeError("persisted D_R input payload is incomplete")
    real_fingerprint = stable_fingerprint(dict(real_payload))
    population_fingerprint = stable_fingerprint(
        dict(population_payload)
    )
    source_cache_fingerprint = population_payload.get(
        "source_cache_fingerprint"
    )
    bounded_cache_fingerprint = population_payload.get(
        "bounded_cache_fingerprint"
    )
    if (
        inputs.get("schema_version")
        != "cure-lite-pacre-v23-D_R-inputs-v1"
        or inputs.get("run_id") != PACRE_VC_DR_RUN_ID
        or inputs.get("authorization_fingerprint")
        != authorization_fingerprint
        or real_payload.get("split") != "D_R"
        or population_payload.get("split") != "D_R"
        or population_payload.get("seed") != 42
        or inputs.get("real_inputs_fingerprint") != real_fingerprint
        or inputs.get("population_fingerprint")
        != population_fingerprint
        or inputs.get("cache_fingerprint")
        != bounded_cache_fingerprint
        or inputs.get("real_inputs_fingerprint")
        != authorization.get("expected_real_inputs_fingerprint")
        or inputs.get("population_fingerprint")
        != authorization.get("expected_population_fingerprint")
        or inputs.get("cache_fingerprint")
        != authorization.get("expected_cache_fingerprint")
        or inputs.get("source_files")
        != authorization.get("expected_real_input_bindings")
        or inputs.get("construction_invocations")
        != {"real_inputs": 1, "population": 1}
        or inputs.get("D_R_accessed") is not True
        or inputs.get("D_V_accessed") is not False
        or inputs.get("D_T_accessed") is not False
        or inputs.get("split_manifest_metadata_read") is not True
        or inputs.get("D_V_tensor_payload_accessed") is not False
        or inputs.get("D_T_tensor_payload_accessed") is not False
        or inputs.get("training_performed") is not False
        or not isinstance(source_cache_fingerprint, str)
        or len(source_cache_fingerprint) != 64
    ):
        raise RuntimeError("persisted D_R input bindings changed")

    source_cache = SimpleNamespace(
        raw_catalog=SimpleNamespace(split="D_R"),
        cache_fingerprint=source_cache_fingerprint,
    )
    real_inputs = SimpleNamespace(
        source_binding=SimpleNamespace(split="D_R"),
        scalar_cache=source_cache,
    )
    bounded_population = SimpleNamespace(
        seed=42,
        source_cache=source_cache,
        source_cache_fingerprint=source_cache_fingerprint,
    )
    return real_inputs, bounded_population


def verify_terminal() -> dict[str, object]:
    output = dr_output_path()
    if (
        not output.is_dir()
        or output.is_symlink()
        or (output / ".incomplete").exists()
        or (output / "FAILURE.json").exists()
    ):
        raise RuntimeError("PACRE-VC D_R output is not terminal")
    live_files = _verify_output_population(output)
    complete = read_strict_json(output / "COMPLETE.json")
    _require_exact_fields(
        complete,
        frozenset(
            {
                "schema_version",
                "run_id",
                "status",
                "authorization_fingerprint",
                "D_R_gate_receipt_fingerprint",
                "D_R_gate_passed",
                "artifact_files",
                "artifact_count",
                "gate_invocations",
                "optimizer_steps",
                "parameter_updates",
                "training_performed",
                "D_R_accessed",
                "D_V_accessed",
                "D_T_accessed",
                "split_manifest_metadata_read",
                "D_V_tensor_payload_accessed",
                "D_T_tensor_payload_accessed",
                "formal_800_route_granted",
                "formal_800_seed",
                "formal_800_epochs",
                "formal_800_steps_per_epoch",
                "formal_800_updates",
                "formal_800_from_scratch",
                "formal_800_execution_authorized",
                "bounded_400_required",
                "bounded_400_authorized",
                "bounded_400_authorization_effect",
                "complete_fingerprint",
            }
        ),
        name="D_R COMPLETE receipt",
    )
    body = dict(complete)
    complete_fingerprint = body.pop("complete_fingerprint", None)
    artifacts = complete.get("artifact_files")
    if (
        not isinstance(complete_fingerprint, str)
        or stable_fingerprint(body) != complete_fingerprint
        or not isinstance(artifacts, dict)
        or complete.get("artifact_count") != len(artifacts)
        or set(artifacts) != EXPECTED_ARTIFACT_FILES
        or complete.get("schema_version")
        != "cure-lite-pacre-v23-D_R-terminal-complete-v1"
        or complete.get("run_id") != PACRE_VC_DR_RUN_ID
        or complete.get("gate_invocations") != 1
        or complete.get("optimizer_steps") != 0
        or complete.get("parameter_updates") != 0
        or complete.get("training_performed") is not False
        or complete.get("D_R_accessed") is not True
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("split_manifest_metadata_read") is not True
        or complete.get("D_V_tensor_payload_accessed") is not False
        or complete.get("D_T_tensor_payload_accessed") is not False
        or complete.get("formal_800_route_granted")
        is not complete.get("D_R_gate_passed")
        or complete.get("formal_800_seed") != 42
        or complete.get("formal_800_epochs") != 800
        or complete.get("formal_800_steps_per_epoch") != 40
        or complete.get("formal_800_updates") != 32000
        or complete.get("formal_800_from_scratch") is not True
        or complete.get("formal_800_execution_authorized") is not False
        or complete.get("bounded_400_required") is not False
        or complete.get("bounded_400_authorized") is not False
        or complete.get("bounded_400_authorization_effect") is not False
    ):
        raise ValueError("PACRE-VC COMPLETE receipt is invalid")
    live_artifacts = {
        name: digest
        for name, digest in live_files.items()
        if name != "COMPLETE.json"
    }
    if live_artifacts != artifacts:
        raise RuntimeError("PACRE-VC terminal artifact population changed")

    authorization = read_strict_json(
        protocol_root() / "D_R_pre_run_authorization.json"
    )
    prerequisites = _read_prerequisites()
    authorization_fingerprint = verify_dr_pre_run_authorization(
        authorization,
        require_output_absent=False,
        **prerequisites,
    )
    attempt = read_strict_json(output / "attempt.json")
    _require_exact_fields(
        attempt,
        frozenset(
            {
                "schema_version",
                "run_id",
                "output_repo_path",
                "authorization_fingerprint",
                "source_closure_fingerprint",
                "runtime_environment_fingerprint",
                "candidate",
                "device",
                "seed",
                "single_attempt",
                "resume_allowed",
                "automatic_retry_allowed",
                "optimizer_allowed",
                "parameter_updates_allowed",
                "D_R_accessed",
                "D_V_accessed",
                "D_T_accessed",
                "split_manifest_metadata_read",
                "D_V_tensor_payload_accessed",
                "D_T_tensor_payload_accessed",
                "training_performed",
                "receipt_fingerprint",
            }
        ),
        name="D_R attempt receipt",
    )
    attempt_fingerprint = verify_fingerprinted(attempt)
    if (
        attempt.get("schema_version")
        != "cure-lite-pacre-v23-D_R-structural-attempt-v1"
        or attempt.get("run_id") != PACRE_VC_DR_RUN_ID
        or attempt.get("output_repo_path")
        != PACRE_VC_DR_OUTPUT_REPO_PATH
        or attempt.get("source_closure_fingerprint")
        != authorization.get("source_closure_fingerprint")
        or attempt.get("runtime_environment_fingerprint")
        != authorization.get("runtime_environment_fingerprint")
        or attempt.get("candidate") != "PACRE-VC-v23"
        or attempt.get("device") != "cuda:0"
        or attempt.get("seed") != 42
        or attempt.get("single_attempt") is not True
        or attempt.get("resume_allowed") is not False
        or attempt.get("automatic_retry_allowed") is not False
        or attempt.get("optimizer_allowed") is not False
        or attempt.get("parameter_updates_allowed") != 0
        or attempt.get("D_R_accessed") is not False
        or attempt.get("D_V_accessed") is not False
        or attempt.get("D_T_accessed") is not False
        or attempt.get("split_manifest_metadata_read") is not False
        or attempt.get("D_V_tensor_payload_accessed") is not False
        or attempt.get("D_T_tensor_payload_accessed") is not False
        or attempt.get("training_performed") is not False
    ):
        raise RuntimeError("PACRE-VC D_R attempt contract changed")

    inputs = read_strict_json(output / "receipts/inputs.json")
    real_inputs, bounded_population = _verify_persisted_input_bindings(
        inputs,
        authorization=authorization,
        authorization_fingerprint=authorization_fingerprint,
    )
    preflight = read_strict_json(output / "receipts/preflight.json")
    _require_exact_fields(
        preflight,
        frozenset(
            {
                "schema_version",
                "run_id",
                "preflight",
                "preflight_fingerprint",
                "training_authorized_by_common_preflight",
                "PACRE_VC_training_authorized",
                "D_R_accessed",
                "D_V_accessed",
                "D_T_accessed",
                "split_manifest_metadata_read",
                "D_V_tensor_payload_accessed",
                "D_T_tensor_payload_accessed",
                "training_performed",
                "receipt_fingerprint",
            }
        ),
        name="D_R bounded preflight receipt",
    )
    preflight_fingerprint = verify_fingerprinted(preflight)
    preflight_payload = preflight.get("preflight")
    preflight_checks = (
        preflight_payload.get("checks")
        if isinstance(preflight_payload, Mapping)
        else None
    )
    if (
        preflight.get("schema_version")
        != "cure-lite-pacre-v23-D_R-bounded-preflight-v1"
        or preflight.get("run_id") != PACRE_VC_DR_RUN_ID
        or not isinstance(preflight_payload, Mapping)
        or preflight.get("preflight_fingerprint")
        != stable_fingerprint(dict(preflight_payload))
        or preflight_payload.get("schema_version")
        != "cure-lite-cslf-bounded-preflight-v1"
        or preflight_payload.get("population_fingerprint")
        != inputs.get("population_fingerprint")
        or preflight_payload.get("bounded_cache_fingerprint")
        != inputs.get("cache_fingerprint")
        or not isinstance(preflight_checks, Mapping)
        or not preflight_checks
        or any(value is not True for value in preflight_checks.values())
        or preflight_payload.get("failed_checks") != []
        or preflight_payload.get("training_authorized") is not True
        or preflight_payload.get("formal_training_authorized") is not False
        or preflight_payload.get("D_V_accessed") is not False
        or preflight_payload.get("D_T_accessed") is not False
        or preflight.get("training_authorized_by_common_preflight")
        is not True
        or preflight.get("PACRE_VC_training_authorized") is not False
        or preflight.get("D_R_accessed") is not True
        or preflight.get("D_V_accessed") is not False
        or preflight.get("D_T_accessed") is not False
        or preflight.get("split_manifest_metadata_read") is not True
        or preflight.get("D_V_tensor_payload_accessed") is not False
        or preflight.get("D_T_tensor_payload_accessed") is not False
        or preflight.get("training_performed") is not False
    ):
        raise RuntimeError("PACRE-VC D_R preflight contract changed")

    wrapper = read_strict_json(output / "receipts/dr_gate.json")
    _require_exact_fields(
        wrapper,
        frozenset(
            {
                "schema_version",
                "run_id",
                "authorization_fingerprint",
                "receipt",
                "receipt_fingerprint",
                "decision",
                "failed_checks",
                "gate_passed",
                "gate_invocations",
                "optimizer_steps",
                "parameter_updates",
                "training_performed",
                "D_R_accessed",
                "D_V_accessed",
                "D_T_accessed",
                "split_manifest_metadata_read",
                "D_V_tensor_payload_accessed",
                "D_T_tensor_payload_accessed",
                "wrapper_fingerprint",
            }
        ),
        name="D_R gate wrapper",
    )
    wrapper_fingerprint = verify_fingerprinted(
        wrapper,
        field="wrapper_fingerprint",
    )
    receipt_payload = wrapper.get("receipt")
    if not isinstance(receipt_payload, dict):
        raise TypeError("persisted D_R receipt is absent")
    receipt = pacre_vc_dr_receipt_from_payload(receipt_payload)
    receipt.verify_sources_unchanged()
    recomputed_checks = recompute_pacre_vc_dr_checks(
        dataset_free_receipt_fingerprint=(
            receipt.dataset_free_receipt_fingerprint
        ),
        real_inputs=real_inputs,
        bounded_population=bounded_population,
        probe=receipt.probe,
    )
    if recomputed_checks != receipt.checks:
        raise RuntimeError("persisted D_R checks do not recompute")
    sealed_replay = receipt.probe.get("sealed_v22_replay_binding")
    sealed_failure = authorization.get("v22_sealed_failure")
    if not isinstance(sealed_replay, Mapping) or not isinstance(
        sealed_failure,
        Mapping,
    ):
        raise RuntimeError("sealed v22 replay association is absent")

    decision = read_strict_json(output / "receipts/decision.json")
    _require_exact_fields(
        decision,
        frozenset(
            {
                "schema_version",
                "run_id",
                "authorization_fingerprint",
                "D_R_gate_receipt_fingerprint",
                "status",
                "failed_checks",
                "D_R_gate_passed",
                "formal_800_route_granted",
                "formal_800_seed",
                "formal_800_epochs",
                "formal_800_steps_per_epoch",
                "formal_800_updates",
                "formal_800_from_scratch",
                "formal_800_execution_authorized",
                "bounded_400_required",
                "bounded_400_authorized",
                "bounded_400_authorization_effect",
                "optimizer_constructed",
                "parameter_updates",
                "training_performed",
                "D_R_accessed",
                "D_V_accessed",
                "D_T_accessed",
                "split_manifest_metadata_read",
                "D_V_tensor_payload_accessed",
                "D_T_tensor_payload_accessed",
                "next_action",
                "receipt_fingerprint",
            }
        ),
        name="D_R terminal decision",
    )
    decision_fingerprint = verify_fingerprinted(decision)
    if (
        attempt.get("authorization_fingerprint")
        != authorization_fingerprint
        or wrapper.get("authorization_fingerprint")
        != authorization_fingerprint
        or receipt.pre_run_authorization_fingerprint
        != authorization_fingerprint
        or receipt.dataset_free_receipt_fingerprint
        != authorization.get("dataset_free_receipt_fingerprint")
        or receipt.runtime_environment_fingerprint
        != authorization.get("runtime_environment_fingerprint")
        or receipt.source_closure_fingerprint
        != authorization.get("source_closure_fingerprint")
        or dict(receipt.prerequisite_fingerprints)
        != authorization.get("prerequisite_fingerprints")
        or receipt.real_inputs_fingerprint
        != inputs.get("real_inputs_fingerprint")
        or receipt.population_fingerprint
        != inputs.get("population_fingerprint")
        or receipt.cache_fingerprint
        != inputs.get("cache_fingerprint")
        or sealed_replay.get(
            "expected_initial_model_fingerprint"
        )
        != sealed_failure.get("initial_model_fingerprint")
        or sealed_replay.get(
            "expected_ordered_target_state_ids_fingerprint"
        )
        != sealed_failure.get(
            "ordered_target_state_ids_fingerprint"
        )
        or wrapper.get("schema_version")
        != "cure-lite-pacre-v23-D_R-gate-wrapper-v1"
        or wrapper.get("run_id") != PACRE_VC_DR_RUN_ID
        or wrapper.get("receipt_fingerprint")
        != receipt.receipt_fingerprint
        or wrapper.get("decision") != receipt.decision
        or wrapper.get("gate_passed") is not receipt.gate_passed
        or wrapper.get("failed_checks") != list(receipt.failed_checks)
        or wrapper.get("gate_invocations") != 1
        or wrapper.get("optimizer_steps") != 0
        or wrapper.get("parameter_updates") != 0
        or wrapper.get("training_performed") is not False
        or wrapper.get("D_R_accessed") is not True
        or wrapper.get("D_V_accessed") is not False
        or wrapper.get("D_T_accessed") is not False
        or wrapper.get("split_manifest_metadata_read") is not True
        or wrapper.get("D_V_tensor_payload_accessed") is not False
        or wrapper.get("D_T_tensor_payload_accessed") is not False
        or decision.get("schema_version")
        != "cure-lite-pacre-v23-D_R-terminal-decision-v1"
        or decision.get("run_id") != PACRE_VC_DR_RUN_ID
        or decision.get("authorization_fingerprint")
        != authorization_fingerprint
        or decision.get("D_R_gate_receipt_fingerprint")
        != receipt.receipt_fingerprint
        or decision.get("status") != receipt.decision
        or decision.get("failed_checks") != list(receipt.failed_checks)
        or decision.get("D_R_gate_passed") is not receipt.gate_passed
        or decision.get("formal_800_route_granted")
        is not receipt.gate_passed
        or decision.get("formal_800_seed") != 42
        or decision.get("formal_800_epochs") != 800
        or decision.get("formal_800_steps_per_epoch") != 40
        or decision.get("formal_800_updates") != 32000
        or decision.get("formal_800_from_scratch") is not True
        or decision.get("formal_800_execution_authorized") is not False
        or decision.get("bounded_400_required") is not False
        or decision.get("bounded_400_authorized") is not False
        or decision.get("bounded_400_authorization_effect") is not False
        or decision.get("optimizer_constructed") is not False
        or decision.get("parameter_updates") != 0
        or decision.get("training_performed") is not False
        or decision.get("D_R_accessed") is not True
        or decision.get("D_V_accessed") is not False
        or decision.get("D_T_accessed") is not False
        or decision.get("split_manifest_metadata_read") is not True
        or decision.get("D_V_tensor_payload_accessed") is not False
        or decision.get("D_T_tensor_payload_accessed") is not False
        or decision.get("next_action")
        != (
            "claim_and_run_unique_formal_800_seed42_from_scratch"
            if receipt.gate_passed
            else "freeze_v23_D_R_negative_and_stop"
        )
        or complete.get("authorization_fingerprint")
        != authorization_fingerprint
        or complete.get("D_R_gate_receipt_fingerprint")
        != receipt.receipt_fingerprint
        or complete.get("status") != receipt.decision
        or complete.get("D_R_gate_passed") is not receipt.gate_passed
    ):
        raise RuntimeError("PACRE-VC terminal associations changed")
    return {
        "run_id": complete["run_id"],
        "output": str(output),
        "decision": receipt.decision,
        "gate_passed": receipt.gate_passed,
        "failed_checks": list(receipt.failed_checks),
        "authorization_fingerprint": authorization_fingerprint,
        "attempt_fingerprint": attempt_fingerprint,
        "inputs_fingerprint": inputs["receipt_fingerprint"],
        "preflight_wrapper_fingerprint": preflight_fingerprint,
        "wrapper_fingerprint": wrapper_fingerprint,
        "receipt_fingerprint": receipt.receipt_fingerprint,
        "decision_fingerprint": decision_fingerprint,
        "complete_fingerprint": complete_fingerprint,
        "artifact_count": complete["artifact_count"],
        "D_R_reopened": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "split_manifest_metadata_read": True,
        "D_V_tensor_payload_accessed": False,
        "D_T_tensor_payload_accessed": False,
        "formal_800_route_granted": receipt.gate_passed,
        "bounded_400_required": False,
        "bounded_400_authorization_effect": False,
        "training_performed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    result = verify_terminal()
    import json

    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
