#!/usr/bin/env python3
"""Consume the sole PACRE-VC authorization and run a read-only D_R gate.

The fixed run constructs the same frozen seed-42 probe population used by the
sealed v22 attempt.  It performs zero optimizer steps and zero parameter
updates.  A terminal 13/13 PASS grants only the route to a separate
from-scratch seed-42 Formal800 authorization.  There is no path to ``D_V``,
``D_T``, bounded training, retry, or resume.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_SEED,
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite_v23.authorization import (
    PACRE_VC_DR_OUTPUT_REPO_PATH,
    PACRE_VC_DR_RUN_ID,
    build_dr_pre_run_authorization,
    dr_output_path,
    frozen_real_dr_source_paths,
    protocol_root,
    verify_dr_pre_run_authorization,
)
from cure_lite_v23.dr_gate import run_pacre_vc_dr_gate
from cure_lite_v23.environment import (
    stabilize_pacre_vc_numerical_runtime,
)
from cure_lite_v23.protocol import (
    fingerprinted,
    read_strict_json,
    write_new_json,
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
AUTHORIZATION_FILE = "D_R_pre_run_authorization.json"
INCOMPLETE_FILE = ".incomplete"


def _read_prerequisites() -> dict[str, dict[str, object]]:
    root = protocol_root()
    return {
        name: read_strict_json(root / filename)
        for name, filename in PREREQUISITE_FILES.items()
    }


def _claim_output(attempt: Mapping[str, object]) -> Path:
    output = dr_output_path()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"single-use D_R output already exists: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / INCOMPLETE_FILE).open("xb").close()
    write_new_json(output / "attempt.json", attempt)
    (output / "receipts").mkdir(exist_ok=False)
    return output


def _artifact_hashes(output: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("D_R artifact may not be a symlink")
        if not path.is_file():
            continue
        relative = str(path.relative_to(output))
        if relative in {
            INCOMPLETE_FILE,
            "COMPLETE.json",
            "FAILURE.json",
        }:
            continue
        rows[relative] = file_sha256(path)
    return rows


def run_once() -> dict[str, object]:
    """Run and terminally seal the one authorized structural gate."""

    stabilize_pacre_vc_numerical_runtime()
    prerequisites = _read_prerequisites()
    authorization_path = protocol_root() / AUTHORIZATION_FILE
    authorization = read_strict_json(authorization_path)
    authorization_fingerprint = verify_dr_pre_run_authorization(
        authorization,
        **prerequisites,
    )
    # Rebuilding explicitly here also establishes that authorization is still
    # the unique live metadata decision immediately before the claim.
    if authorization != build_dr_pre_run_authorization(**prerequisites):
        raise RuntimeError("D_R authorization differs before output claim")
    attempt = fingerprinted(
        {
            "schema_version": (
                "cure-lite-pacre-v23-D_R-structural-attempt-v1"
            ),
            "run_id": PACRE_VC_DR_RUN_ID,
            "output_repo_path": PACRE_VC_DR_OUTPUT_REPO_PATH,
            "authorization_fingerprint": authorization_fingerprint,
            "source_closure_fingerprint": (
                authorization["source_closure_fingerprint"]
            ),
            "runtime_environment_fingerprint": (
                authorization["runtime_environment_fingerprint"]
            ),
            "candidate": "PACRE-VC-v23",
            "device": "cuda:0",
            "seed": COVERAGE_STATE_BOUNDED_SEED,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "optimizer_allowed": False,
            "parameter_updates_allowed": 0,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "split_manifest_metadata_read": False,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "training_performed": False,
        }
    )
    output = _claim_output(attempt)
    receipts = output / "receipts"
    try:
        source_paths = frozen_real_dr_source_paths()
        real_inputs = build_coverage_state_real_dr_inputs(**source_paths)
        population = build_coverage_state_bounded_population(
            real_inputs.scalar_cache,
            seed=COVERAGE_STATE_BOUNDED_SEED,
        )
        preflight = prepare_coverage_state_bounded_preflight(population)
        if (
            real_inputs.build_fingerprint
            != authorization["expected_real_inputs_fingerprint"]
            or population.population_fingerprint
            != authorization["expected_population_fingerprint"]
            or population.cache.cache_fingerprint
            != authorization["expected_cache_fingerprint"]
            or not preflight.training_authorized
        ):
            raise RuntimeError(
                "live D_R graph differs from the sealed v22 binding"
            )
        inputs_receipt = fingerprinted(
            {
                "schema_version": (
                    "cure-lite-pacre-v23-D_R-inputs-v1"
                ),
                "run_id": PACRE_VC_DR_RUN_ID,
                "authorization_fingerprint": (
                    authorization_fingerprint
                ),
                "real_inputs": real_inputs.canonical_payload(),
                "real_inputs_fingerprint": (
                    real_inputs.build_fingerprint
                ),
                "population": population.canonical_payload(),
                "population_fingerprint": (
                    population.population_fingerprint
                ),
                "cache_fingerprint": (
                    population.cache.cache_fingerprint
                ),
                "source_files": {
                    name: {
                        "repo_path": str(
                            path.relative_to(Path(__file__).resolve().parents[1])
                        ),
                        "file_sha256": file_sha256(path),
                    }
                    for name, path in sorted(source_paths.items())
                },
                "construction_invocations": {
                    "real_inputs": 1,
                    "population": 1,
                },
                "D_R_accessed": True,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "split_manifest_metadata_read": True,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
                "training_performed": False,
            }
        )
        write_new_json(receipts / "inputs.json", inputs_receipt)
        preflight_receipt = fingerprinted(
            {
                "schema_version": (
                    "cure-lite-pacre-v23-D_R-bounded-preflight-v1"
                ),
                "run_id": PACRE_VC_DR_RUN_ID,
                "preflight": preflight.canonical_payload(),
                "preflight_fingerprint": (
                    preflight.preflight_fingerprint
                ),
                "training_authorized_by_common_preflight": (
                    preflight.training_authorized
                ),
                "PACRE_VC_training_authorized": False,
                "D_R_accessed": True,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "split_manifest_metadata_read": True,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
                "training_performed": False,
            }
        )
        write_new_json(receipts / "preflight.json", preflight_receipt)

        gate = run_pacre_vc_dr_gate(
            dataset_free_receipt=prerequisites["dataset_free"],
            pre_run_authorization=authorization,
            runtime_environment_lock=prerequisites[
                "runtime_selected_device"
            ],
            real_inputs=real_inputs,
            bounded_population=population,
            device="cuda:0",
        )
        gate_payload = gate.canonical_payload()
        gate_wrapper = fingerprinted(
            {
                "schema_version": (
                    "cure-lite-pacre-v23-D_R-gate-wrapper-v1"
                ),
                "run_id": PACRE_VC_DR_RUN_ID,
                "authorization_fingerprint": (
                    authorization_fingerprint
                ),
                "receipt": gate_payload,
                "receipt_fingerprint": gate.receipt_fingerprint,
                "decision": gate.decision,
                "failed_checks": list(gate.failed_checks),
                "gate_passed": gate.gate_passed,
                "gate_invocations": 1,
                "optimizer_steps": 0,
                "parameter_updates": 0,
                "training_performed": False,
                "D_R_accessed": True,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "split_manifest_metadata_read": True,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
            },
            field="wrapper_fingerprint",
        )
        write_new_json(receipts / "dr_gate.json", gate_wrapper)
        decision = fingerprinted(
            {
                "schema_version": (
                    "cure-lite-pacre-v23-D_R-terminal-decision-v1"
                ),
                "run_id": PACRE_VC_DR_RUN_ID,
                "authorization_fingerprint": (
                    authorization_fingerprint
                ),
                "D_R_gate_receipt_fingerprint": (
                    gate.receipt_fingerprint
                ),
                "status": gate.decision,
                "failed_checks": list(gate.failed_checks),
                "D_R_gate_passed": gate.gate_passed,
                "formal_800_route_granted": gate.gate_passed,
                "formal_800_seed": 42,
                "formal_800_epochs": 800,
                "formal_800_steps_per_epoch": 40,
                "formal_800_updates": 32000,
                "formal_800_from_scratch": True,
                "formal_800_execution_authorized": False,
                "bounded_400_required": False,
                "bounded_400_authorized": False,
                "bounded_400_authorization_effect": False,
                "optimizer_constructed": False,
                "parameter_updates": 0,
                "training_performed": False,
                "D_R_accessed": True,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "split_manifest_metadata_read": True,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
                "next_action": (
                    "claim_and_run_unique_formal_800_seed42_from_scratch"
                    if gate.gate_passed
                    else "freeze_v23_D_R_negative_and_stop"
                ),
            }
        )
        write_new_json(receipts / "decision.json", decision)
        artifact_hashes = _artifact_hashes(output)
        complete_body: dict[str, object] = {
            "schema_version": (
                "cure-lite-pacre-v23-D_R-terminal-complete-v1"
            ),
            "run_id": PACRE_VC_DR_RUN_ID,
            "status": gate.decision,
            "authorization_fingerprint": authorization_fingerprint,
            "D_R_gate_receipt_fingerprint": gate.receipt_fingerprint,
            "D_R_gate_passed": gate.gate_passed,
            "artifact_files": artifact_hashes,
            "artifact_count": len(artifact_hashes),
            "gate_invocations": 1,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "training_performed": False,
            "D_R_accessed": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "split_manifest_metadata_read": True,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "formal_800_route_granted": gate.gate_passed,
            "formal_800_seed": 42,
            "formal_800_epochs": 800,
            "formal_800_steps_per_epoch": 40,
            "formal_800_updates": 32000,
            "formal_800_from_scratch": True,
            "formal_800_execution_authorized": False,
            "bounded_400_required": False,
            "bounded_400_authorized": False,
            "bounded_400_authorization_effect": False,
        }
        complete = {
            **complete_body,
            "complete_fingerprint": stable_fingerprint(complete_body),
        }
        write_new_json(output / "COMPLETE.json", complete)
        (output / INCOMPLETE_FILE).unlink()
        return {
            "run_id": PACRE_VC_DR_RUN_ID,
            "output": str(output),
            "decision": gate.decision,
            "failed_checks": list(gate.failed_checks),
            "gate_passed": gate.gate_passed,
            "formal_800_route_granted": gate.gate_passed,
            "complete_fingerprint": complete["complete_fingerprint"],
            "D_V_accessed": False,
            "D_T_accessed": False,
            "split_manifest_metadata_read": True,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
            "bounded_400_required": False,
            "bounded_400_authorization_effect": False,
        }
    except BaseException as error:
        failure = fingerprinted(
            {
                "schema_version": (
                    "cure-lite-pacre-v23-D_R-incomplete-failure-v1"
                ),
                "run_id": PACRE_VC_DR_RUN_ID,
                "exception_type": type(error).__name__,
                "message": str(error),
                "authorization_fingerprint": (
                    authorization_fingerprint
                ),
                "artifact_files_before_failure": (
                    _artifact_hashes(output)
                ),
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "split_manifest_metadata_may_have_been_read": True,
                "D_V_tensor_payload_accessed": False,
                "D_T_tensor_payload_accessed": False,
            }
        )
        try:
            write_new_json(output / "FAILURE.json", failure)
        except BaseException:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-once",
        action="store_true",
        required=True,
        help="consume the fixed authorization and run the D_R gate once",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    result = run_once()
    import json

    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
