"""Fail-closed pre-run authorization for PACRE-VC v23.

This module handles metadata only.  It verifies generated receipts, the
sealed v22 failure, source/runtime locks, and frozen input-file hashes before
authorizing one read-only ``D_R`` gate.  It never constructs or loads a real
data tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping

from cure_lite.cache.schema import file_sha256, stable_fingerprint

from .dataset_free import run_pacre_vc_dataset_free_gate
from .environment import verify_runtime_environment
from .pacre_vc import PACRE_VC_CANDIDATE
from .protocol import (
    read_strict_json,
    repository_root,
    source_closure_payload,
    verify_fingerprinted,
    verify_source_closure,
)


PACRE_VC_PROTOCOL_REPO_PATH: Final = (
    "protocols/IRSTD-1K/pacre_v23_verifier_corrected"
)
PACRE_VC_DR_OUTPUT_REPO_PATH: Final = (
    "runs/irstd1k_stage_a_seed42/"
    "pacre_v23_verifier_corrected_D_R_structural_r1"
)
PACRE_VC_DR_RUN_ID: Final = (
    "pacre_v23_verifier_corrected_D_R_structural_r1"
)
PACRE_VC_DR_AUTHORIZATION_SCHEMA: Final = (
    "cure-lite-pacre-v23-D_R-pre-run-authorization-v1"
)
PACRE_VC_V22_FAILURE_INHERITANCE_SCHEMA: Final = (
    "cure-lite-pacre-v23-v22-failure-inheritance-v1"
)
PACRE_VC_RUNNER_VERIFICATION_SCHEMA: Final = (
    "cure-lite-pacre-v23-runner-verification-v1"
)
PACRE_VC_DESIGN_REPO_PATH: Final = (
    f"{PACRE_VC_PROTOCOL_REPO_PATH}/verifier_design_preregistration.md"
)
PACRE_VC_V22_OUTPUT_REPO_PATH: Final = (
    "runs/irstd1k_stage_a_seed42/"
    "cure_lite_pacre_v22_pmope_bounded_400_seed42_r1"
)
PACRE_VC_V22_ARTIFACTS: Final = (
    (
        "attempt",
        f"{PACRE_VC_V22_OUTPUT_REPO_PATH}/attempt.json",
        "53bfa290edede646dc50091a31d0ce59fd52e3705779876b8cfc58b8176110ab",
    ),
    (
        "config",
        f"{PACRE_VC_V22_OUTPUT_REPO_PATH}/receipts/config.json",
        "9ba0f0f4bb0186cb785a6673569f9a4485cd0129847f0d1d0135b0be181d7235",
    ),
    (
        "dataset_free",
        f"{PACRE_VC_V22_OUTPUT_REPO_PATH}/receipts/dataset_free.json",
        "f389c99df801a0008c2ee84110a147789ec2327a1c6f5717e58d55f96ba9d715",
    ),
    (
        "dr_gate",
        f"{PACRE_VC_V22_OUTPUT_REPO_PATH}/receipts/dr_gate.json",
        "ff0b946577f4207a4a38e257db87ab990eb4ce1142db916e94705f036f6d4b2a",
    ),
    (
        "inputs",
        f"{PACRE_VC_V22_OUTPUT_REPO_PATH}/receipts/inputs.json",
        "e7968862f5f10ec3a6fcf236a40e4f4a50fd194e047c9cf59d11ca033e1d29b7",
    ),
    (
        "preflight",
        f"{PACRE_VC_V22_OUTPUT_REPO_PATH}/receipts/preflight.json",
        "41cf86cdc9b6e258d08fc203bbf06bb9ede9824d70d2e3df3bed20f029dcde2e",
    ),
    (
        "decision",
        f"{PACRE_VC_V22_OUTPUT_REPO_PATH}/receipts/decision.json",
        "0e1f4b53ccc737caaceff79ed227f269fa142c28507360ef9641f2ff2667cd06",
    ),
    (
        "complete",
        f"{PACRE_VC_V22_OUTPUT_REPO_PATH}/COMPLETE.json",
        "fc9677521678653db4481e8136a47ee525748ff8eb9c223d1d91fff87e577cc3",
    ),
)
PACRE_VC_FROZEN_REAL_DR_INPUTS: Final = (
    (
        "manifest_path",
        "protocols/IRSTD-1K/stage_a_seed42/manifest.json",
        "aa8e33529bd86f564ce6e163e0f9a7b1b3053e9c15054a59c6702a1523f35c02",
    ),
    (
        "state_index_path",
        (
            "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3/"
            "d_r/state_cache/index.json"
        ),
        "075fc1ad217f365df85b1d29568ad215f06ce6e0b691ef78a5dd85f0affe6298",
    ),
    (
        "geometry_config_path",
        "protocols/IRSTD-1K/geometry_safe_p0_v2/config.json",
        "719e956b7c51b2b2c8294699fe26c2d36d5c8190b0d8bb5c1d5665a0f4344558",
    ),
    (
        "geometry_receipt_path",
        (
            "runs/irstd1k_stage_a_seed42/"
            "cure_lite_geometry_safe_p0_v2_r1/"
            "receipts/geometry_catalog.json"
        ),
        "e2a9a986f8819433f3f5efd5c4f627504d10fb32d20f62769b2235b803209283",
    ),
    (
        "observability_config_path",
        (
            "protocols/IRSTD-1K/"
            "coverage_state_observability_v1/config.json"
        ),
        "60d42e657f1daed3cb01c7ee93c8f3fe17417542931d853756ccbbeda1f95713",
    ),
)
PACRE_VC_V22_DR_RECEIPT_FINGERPRINT: Final = (
    "2f71f76452e7c35c80c2394047939bb6ebd21a9aed7962de591a71475450f027"
)
PACRE_VC_V22_COMPLETE_FINGERPRINT: Final = (
    "0f71dce1f48982b6dd2e559cea113f2170e4bc2c04fcdcf3557c61df51e9da74"
)
PACRE_VC_V22_POPULATION_FINGERPRINT: Final = (
    "1a53467d57bea595afcc1edd3330708d1dda39e0e2d606325e552e8993e7841c"
)
PACRE_VC_V22_CACHE_FINGERPRINT: Final = (
    "c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca00754c206092aaf66d8"
)
PACRE_VC_V22_REAL_INPUTS_FINGERPRINT: Final = (
    "ee717a7e13461fb86cacc65d33efd331abcf9b27611f254f981082d45eb7bfb4"
)


def protocol_root() -> Path:
    return repository_root() / PACRE_VC_PROTOCOL_REPO_PATH


def dr_output_path() -> Path:
    return repository_root() / PACRE_VC_DR_OUTPUT_REPO_PATH


def _canonical_repo_file(relative: str) -> Path:
    root = repository_root()
    path = root / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError(f"invalid frozen repository file: {relative}")
    return path


def frozen_real_dr_source_paths() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, relative, expected in PACRE_VC_FROZEN_REAL_DR_INPUTS:
        path = _canonical_repo_file(relative)
        if file_sha256(path) != expected:
            raise RuntimeError(f"frozen real D_R input changed: {name}")
        result[name] = path
    return result


def _receipt_fingerprint(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> str:
    return verify_fingerprinted(payload, field=field)


def build_v22_failure_inheritance_receipt() -> dict[str, object]:
    """Revalidate the sealed v22 terminal metadata without loading tensors."""

    artifacts: dict[str, dict[str, str]] = {}
    parsed: dict[str, dict[str, object]] = {}
    for name, relative, expected_sha in PACRE_VC_V22_ARTIFACTS:
        path = _canonical_repo_file(relative)
        observed_sha = file_sha256(path)
        if observed_sha != expected_sha:
            raise RuntimeError(f"sealed v22 artifact changed: {name}")
        artifacts[name] = {
            "repo_path": relative,
            "file_sha256": observed_sha,
        }
        parsed[name] = read_strict_json(path)

    dr_wrapper = parsed["dr_gate"]
    dr_receipt = dr_wrapper.get("receipt")
    inputs = parsed["inputs"]
    complete = parsed["complete"]
    decision = parsed["decision"]
    complete_body = dict(complete)
    observed_complete_fingerprint = complete_body.pop(
        "complete_fingerprint",
        None,
    )
    expected_v22_artifacts = {
        relative.removeprefix(
            f"{PACRE_VC_V22_OUTPUT_REPO_PATH}/"
        ): expected_sha
        for name, relative, expected_sha in PACRE_VC_V22_ARTIFACTS
        if name != "complete"
    }
    if (
        not isinstance(dr_receipt, Mapping)
        or dr_wrapper.get("receipt_fingerprint")
        != PACRE_VC_V22_DR_RECEIPT_FINGERPRINT
        or stable_fingerprint(dict(dr_receipt))
        != PACRE_VC_V22_DR_RECEIPT_FINGERPRINT
        or dr_wrapper.get("decision")
        != "PACRE_V22_D_R_STRUCTURAL_FAIL"
        or dr_wrapper.get("gate_passed") is not False
        or dr_wrapper.get("failed_checks")
        != ["05_phase_residual_and_compatibility_algebra_valid"]
        or inputs.get("population_fingerprint")
        != PACRE_VC_V22_POPULATION_FINGERPRINT
        or inputs.get("bounded_cache_fingerprint")
        != PACRE_VC_V22_CACHE_FINGERPRINT
        or inputs.get("real_inputs_fingerprint")
        != PACRE_VC_V22_REAL_INPUTS_FINGERPRINT
        or complete.get("complete_fingerprint")
        != PACRE_VC_V22_COMPLETE_FINGERPRINT
        or observed_complete_fingerprint
        != stable_fingerprint(complete_body)
        or complete.get("artifact_files") != expected_v22_artifacts
        or complete.get("artifact_file_count")
        != len(expected_v22_artifacts)
        or complete.get("decision") != "PACRE_V22_D_R_GATE_FAIL"
        or decision.get("status") != "PACRE_V22_D_R_GATE_FAIL"
        or decision.get("bounded_training_performed") is not False
        or decision.get("D_V_accessed") is not False
        or decision.get("D_T_accessed") is not False
    ):
        raise RuntimeError("sealed v22 failure contract changed")
    v22_probe = dr_receipt.get("probe")
    representation = (
        v22_probe.get("representation")
        if isinstance(v22_probe, Mapping)
        else None
    )
    target_rows = (
        representation.get("target_rows")
        if isinstance(representation, Mapping)
        else None
    )
    if (
        not isinstance(v22_probe, Mapping)
        or not isinstance(target_rows, list)
        or len(target_rows) != 32
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("state_id"), str)
            for row in target_rows
        )
        or not isinstance(
            v22_probe.get("initial_model_fingerprint"),
            str,
        )
    ):
        raise RuntimeError("sealed v22 replay binding is incomplete")
    target_state_ids = [
        str(row["state_id"]) for row in target_rows
    ]
    body: dict[str, object] = {
        "schema_version": PACRE_VC_V22_FAILURE_INHERITANCE_SCHEMA,
        "candidate": PACRE_VC_CANDIDATE,
        "v22_candidate": "PACRE-v22",
        "v22_decision": "PACRE_V22_D_R_STRUCTURAL_FAIL",
        "v22_failed_checks": [
            "05_phase_residual_and_compatibility_algebra_valid"
        ],
        "v22_D_R_receipt_fingerprint": (
            PACRE_VC_V22_DR_RECEIPT_FINGERPRINT
        ),
        "v22_complete_fingerprint": PACRE_VC_V22_COMPLETE_FINGERPRINT,
        "v22_population_fingerprint": (
            PACRE_VC_V22_POPULATION_FINGERPRINT
        ),
        "v22_cache_fingerprint": PACRE_VC_V22_CACHE_FINGERPRINT,
        "v22_real_inputs_fingerprint": (
            PACRE_VC_V22_REAL_INPUTS_FINGERPRINT
        ),
        "v22_selected_device": "cuda:0",
        "v22_execution_seed": 42,
        "v22_initial_model_fingerprint": (
            v22_probe["initial_model_fingerprint"]
        ),
        "v22_ordered_target_state_ids": target_state_ids,
        "v22_ordered_target_state_ids_fingerprint": (
            stable_fingerprint(target_state_ids)
        ),
        "sealed_artifacts": artifacts,
        "adaptive_attempt": True,
        "independent_confirmation": False,
        "v22_receipt_rewritten": False,
        "historical_D_R_receipt_metadata_read": True,
        "D_R_cached_tensor_payload_accessed": False,
        "new_D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    return {
        **body,
        "receipt_fingerprint": stable_fingerprint(body),
    }


def build_runner_verification_receipt() -> dict[str, object]:
    """Bind all CLI entrypoints before any real-data authorization."""

    required = (
        "tools/run_cure_lite_v23_pacre_vc_preflight.py",
        "tools/run_cure_lite_v23_pacre_vc_dr_gate.py",
        "tools/verify_cure_lite_v23_pacre_vc_dr_receipt.py",
        "tools/run_cure_lite_v23_pacre_vc_formal_800.py",
        (
            "tools/"
            "verify_cure_lite_v23_pacre_vc_formal_800_receipt.py"
        ),
        "tools/run_cure_lite_v23_pacre_vc_formal_d_v.py",
        (
            "tools/"
            "verify_cure_lite_v23_pacre_vc_formal_d_v_receipt.py"
        ),
    )
    files = {
        relative: file_sha256(_canonical_repo_file(relative))
        for relative in required
    }
    body: dict[str, object] = {
        "schema_version": PACRE_VC_RUNNER_VERIFICATION_SCHEMA,
        "candidate": PACRE_VC_CANDIDATE,
        "required_entrypoints": list(required),
        "files": files,
        "create_only_artifacts": True,
        "D_R_runner_requires_authorization": True,
        "D_R_runner_zero_update": True,
        "formal_800_runner_requires_terminal_D_R_pass": True,
        "formal_800_seed": 42,
        "formal_800_epochs": 800,
        "formal_800_steps_per_epoch": 40,
        "formal_800_updates": 32000,
        "formal_800_from_scratch": True,
        "bounded_400_required": False,
        "bounded_400_authorization_effect": False,
        "D_V_entrypoint_present": True,
        "D_V_requires_verified_formal_800_terminal": True,
        "D_V_adaptive": True,
        "D_V_fixed_device": "cuda:0",
        "D_V_batch_size": 8,
        "D_V_best_base_strict_improvement": True,
        "D_V_minimum_fixed_uplift_margin": None,
        "D_V_plus_one_is_sufficient": True,
        "D_V_authorizes_D_T": False,
        "D_T_entrypoint_present": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    return {
        **body,
        "receipt_fingerprint": stable_fingerprint(body),
    }


def _validate_v22_inheritance(payload: Mapping[str, object]) -> str:
    fingerprint = _receipt_fingerprint(payload)
    current = build_v22_failure_inheritance_receipt()
    if dict(payload) != current:
        raise RuntimeError("v22 failure inheritance receipt changed")
    return fingerprint


def _validate_runner(payload: Mapping[str, object]) -> str:
    fingerprint = _receipt_fingerprint(payload)
    current = build_runner_verification_receipt()
    if dict(payload) != current:
        raise RuntimeError("runner verification receipt changed")
    return fingerprint


def build_dr_pre_run_authorization(
    *,
    v22_failure_inheritance: Mapping[str, object],
    runtime_cpu: Mapping[str, object],
    runtime_selected_device: Mapping[str, object],
    forward_parity: Mapping[str, object],
    scalar_counterexample: Mapping[str, object],
    cpu_stress: Mapping[str, object],
    selected_device_stress: Mapping[str, object],
    dataset_free: Mapping[str, object],
    implementation_closure: Mapping[str, object],
    runner_verification: Mapping[str, object],
    require_output_absent: bool = True,
) -> dict[str, object]:
    """Build the sole metadata authorization for the v23 ``D_R`` gate."""

    if type(require_output_absent) is not bool:
        raise TypeError("require_output_absent must be bool")
    if require_output_absent and (
        dr_output_path().exists() or dr_output_path().is_symlink()
    ):
        raise FileExistsError(
            "PACRE-VC D_R result directory already exists"
        )
    frozen_paths = frozen_real_dr_source_paths()
    design = _canonical_repo_file(PACRE_VC_DESIGN_REPO_PATH)
    v22_fingerprint = _validate_v22_inheritance(
        v22_failure_inheritance
    )
    runner_fingerprint = _validate_runner(runner_verification)
    source_fingerprint = verify_source_closure(
        implementation_closure
    )
    current_source = source_closure_payload()
    if dict(implementation_closure) != current_source:
        raise RuntimeError("implementation closure is not current")
    cpu_environment_fingerprint = verify_runtime_environment(
        runtime_cpu,
        "cpu",
    )
    selected_environment_fingerprint = verify_runtime_environment(
        runtime_selected_device,
        "cuda:0",
    )
    parity_fingerprint = _receipt_fingerprint(forward_parity)
    counterexample_fingerprint = _receipt_fingerprint(
        scalar_counterexample
    )
    cpu_stress_fingerprint = _receipt_fingerprint(cpu_stress)
    selected_stress_fingerprint = _receipt_fingerprint(
        selected_device_stress
    )
    dataset_fingerprint = _receipt_fingerprint(dataset_free)
    recomputed_dataset_free = run_pacre_vc_dataset_free_gate(
        parity_receipt=forward_parity,
        cpu_stress_receipt=cpu_stress,
        selected_device_stress_receipt=selected_device_stress,
        counterexample_receipt=scalar_counterexample,
        runtime_environment_receipts={
            "cpu": runtime_cpu,
            "cuda:0": runtime_selected_device,
        },
        source_closure_receipt=implementation_closure,
    )
    if dict(dataset_free) != recomputed_dataset_free:
        raise PermissionError(
            "PACRE-VC dataset-free receipt does not independently recompute"
        )
    evidence = dataset_free.get("evidence_bindings")
    if (
        dataset_free.get("candidate") != PACRE_VC_CANDIDATE
        or dataset_free.get("gate_passed") is not True
        or dataset_free.get("D_R_accessed") is not False
        or dataset_free.get("D_V_accessed") is not False
        or dataset_free.get("D_T_accessed") is not False
        or dataset_free.get("training_performed") is not False
        or not isinstance(evidence, Mapping)
        or evidence.get("parity_receipt_fingerprint")
        != parity_fingerprint
        or evidence.get("counterexample_receipt_fingerprint")
        != counterexample_fingerprint
        or evidence.get("cpu_stress_receipt_fingerprint")
        != cpu_stress_fingerprint
        or evidence.get(
            "selected_device_stress_receipt_fingerprint"
        )
        != selected_stress_fingerprint
        or evidence.get("cpu_runtime_environment_fingerprint")
        != cpu_environment_fingerprint
        or evidence.get("selected_runtime_environment_fingerprint")
        != selected_environment_fingerprint
        or evidence.get("source_closure_fingerprint")
        != source_fingerprint
    ):
        raise PermissionError(
            "PACRE-VC dataset-free evidence binding is invalid"
        )
    prerequisite_fingerprints = {
        "cpu_runtime_environment": cpu_environment_fingerprint,
        "cpu_stress": cpu_stress_fingerprint,
        "dataset_free": dataset_fingerprint,
        "forward_parity": parity_fingerprint,
        "implementation_closure": source_fingerprint,
        "runner_verification": runner_fingerprint,
        "scalar_counterexample": counterexample_fingerprint,
        "selected_device_runtime_environment": (
            selected_environment_fingerprint
        ),
        "selected_device_stress": selected_stress_fingerprint,
        "v22_failure_inheritance": v22_fingerprint,
    }
    body: dict[str, object] = {
        "schema_version": PACRE_VC_DR_AUTHORIZATION_SCHEMA,
        "run_id": PACRE_VC_DR_RUN_ID,
        "candidate": PACRE_VC_CANDIDATE,
        "status": "PACRE_V23_D_R_PRE_RUN_AUTHORIZED",
        "adaptive_attempt": True,
        "independent_confirmation": False,
        "source_closure_fingerprint": source_fingerprint,
        "runtime_environment_fingerprint": (
            selected_environment_fingerprint
        ),
        "cpu_runtime_environment_fingerprint": (
            cpu_environment_fingerprint
        ),
        "dataset_free_receipt_fingerprint": dataset_fingerprint,
        "prerequisite_fingerprints": prerequisite_fingerprints,
        "v22_sealed_failure": {
            "receipt_fingerprint": (
                PACRE_VC_V22_DR_RECEIPT_FINGERPRINT
            ),
            "complete_fingerprint": (
                PACRE_VC_V22_COMPLETE_FINGERPRINT
            ),
            "decision": "PACRE_V22_D_R_STRUCTURAL_FAIL",
            "failed_checks": [
                "05_phase_residual_and_compatibility_algebra_valid"
            ],
            "initial_model_fingerprint": (
                v22_failure_inheritance[
                    "v22_initial_model_fingerprint"
                ]
            ),
            "ordered_target_state_ids_fingerprint": (
                v22_failure_inheritance[
                    "v22_ordered_target_state_ids_fingerprint"
                ]
            ),
        },
        "expected_real_input_bindings": {
            name: {
                "repo_path": str(path.relative_to(repository_root())),
                "file_sha256": expected,
            }
            for (name, _, expected), path in zip(
                PACRE_VC_FROZEN_REAL_DR_INPUTS,
                (frozen_paths[name] for name, _, _ in PACRE_VC_FROZEN_REAL_DR_INPUTS),
                strict=True,
            )
        },
        "expected_real_inputs_fingerprint": (
            PACRE_VC_V22_REAL_INPUTS_FINGERPRINT
        ),
        "expected_population_fingerprint": (
            PACRE_VC_V22_POPULATION_FINGERPRINT
        ),
        "expected_cache_fingerprint": (
            PACRE_VC_V22_CACHE_FINGERPRINT
        ),
        "selected_device": "cuda:0",
        "execution_seed": 42,
        "target_state_count": 32,
        "context_state_count": 96,
        "design_repo_path": PACRE_VC_DESIGN_REPO_PATH,
        "design_file_sha256": file_sha256(design),
        "D_R_output_repo_path": PACRE_VC_DR_OUTPUT_REPO_PATH,
        "D_R_output_absent_at_authorization": True,
        "single_use": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "optimizer_allowed": False,
        "parameter_updates_allowed": 0,
        "formal_800_route_granted_by_D_R_pass_only": True,
        "formal_800_seed": 42,
        "formal_800_epochs": 800,
        "formal_800_steps_per_epoch": 40,
        "formal_800_updates": 32000,
        "formal_800_from_scratch": True,
        "bounded_400_required": False,
        "bounded_400_authorized": False,
        "bounded_400_authorization_effect": False,
        "formal_800_execution_authorized": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "split_manifest_file_hashed": True,
        "split_manifest_metadata_read": False,
        "D_V_tensor_payload_accessed": False,
        "D_T_tensor_payload_accessed": False,
        "training_performed": False,
    }
    return {
        **body,
        "authorization_fingerprint": stable_fingerprint(body),
    }


def verify_dr_pre_run_authorization(
    authorization: Mapping[str, object],
    *,
    require_output_absent: bool = True,
    **receipts: Mapping[str, object],
) -> str:
    """Rebuild and require byte-level canonical authorization equality."""

    fingerprint = verify_fingerprinted(
        authorization,
        field="authorization_fingerprint",
    )
    expected = build_dr_pre_run_authorization(
        **receipts,
        require_output_absent=require_output_absent,
    )
    if dict(authorization) != expected:
        raise RuntimeError("PACRE-VC D_R authorization changed")
    return fingerprint


__all__ = [
    "PACRE_VC_DESIGN_REPO_PATH",
    "PACRE_VC_DR_AUTHORIZATION_SCHEMA",
    "PACRE_VC_DR_OUTPUT_REPO_PATH",
    "PACRE_VC_DR_RUN_ID",
    "PACRE_VC_FROZEN_REAL_DR_INPUTS",
    "PACRE_VC_PROTOCOL_REPO_PATH",
    "PACRE_VC_RUNNER_VERIFICATION_SCHEMA",
    "PACRE_VC_V22_FAILURE_INHERITANCE_SCHEMA",
    "build_dr_pre_run_authorization",
    "build_runner_verification_receipt",
    "build_v22_failure_inheritance_receipt",
    "dr_output_path",
    "frozen_real_dr_source_paths",
    "protocol_root",
    "verify_dr_pre_run_authorization",
]
