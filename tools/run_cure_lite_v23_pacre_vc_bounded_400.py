#!/usr/bin/env python3
"""Run the unique PACRE-VC seed-42 bounded-400 attempt after D_R PASS.

The command is fixed to CUDA logical device 0, one fresh Adam optimizer,
10 epochs by 40 updates, PMOPE, and threshold zero.  It has no retry, resume,
``D_V``, ``D_T``, calibration, or Formal-800 option.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_EPOCHS,
    COVERAGE_STATE_BOUNDED_SEED,
    COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH,
    COVERAGE_STATE_BOUNDED_UPDATES,
    build_coverage_state_bounded_population,
    prepare_coverage_state_bounded_preflight,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite_v23.authorization import (
    frozen_real_dr_source_paths,
    protocol_root,
)
from cure_lite_v23.bounded_runner import (
    PACRE_BOUNDED_ATTEMPT_RECEIPT_SCHEMA,
    PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG,
    PACRE_BOUNDED_DEVICE,
    PACRE_BOUNDED_OUTPUT_PATH,
    PACRE_BOUNDED_OUTPUT_REPO_PATH,
    PACRE_BOUNDED_PAUSE_TEMPERATURE_C,
    PACRE_BOUNDED_RESUME_TEMPERATURE_C,
    PACRE_BOUNDED_RUN_ID,
    PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256,
    PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH,
    PACRE_BOUNDED_VISIBLE_GPU,
    load_pacre_bounded_output_claim,
    pacre_bounded_process_identity,
    prepare_pacre_bounded_run_authorization,
    run_pacre_pmope_bounded_400,
)
from cure_lite_v23.dataset_free import (
    PACRE_FORMAL_FEATURE_CHANNELS,
    PACRE_FORMAL_FEATURE_STRIDE,
    PACRE_FORMAL_PARAMETER_COUNT,
    PACRE_FORMAL_WIDTH,
)
from cure_lite_v23.dr_gate import (
    PACRE_VC_DR_PASS_DECISION,
    pacre_vc_dr_receipt_from_payload,
)
from cure_lite_v23.environment import (
    stabilize_pacre_vc_numerical_runtime,
)
from cure_lite_v23.pacre_vc import (
    PACRE_VC_CANDIDATE,
    CoverageStatePACREVerifierCorrectedConfig,
)
from cure_lite_v23.protocol import (
    fingerprinted,
    read_strict_json,
    write_new_json,
)
from cure_lite_v23.training import PACRE_PMOPE_OBJECTIVE
from tools.verify_cure_lite_v23_pacre_vc_dr_receipt import (
    verify_terminal as verify_dr_terminal,
)


INCOMPLETE_FILE = ".incomplete"
ROOT = Path(__file__).resolve().parents[1]


def _flag_value(tokens: Sequence[str], name: str) -> str:
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == name:
            if index + 1 >= len(tokens):
                raise RuntimeError(f"temperature wrapper lacks {name}")
            values.append(tokens[index + 1])
        elif token.startswith(f"{name}="):
            values.append(token.split("=", maxsplit=1)[1])
    if len(values) != 1:
        raise RuntimeError(
            f"temperature wrapper must specify {name} exactly once"
        )
    return values[0]


def _runtime_contract() -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PACRE_BOUNDED_VISIBLE_GPU:
        raise RuntimeError("bounded-400 fixes CUDA_VISIBLE_DEVICES=0")
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "bounded-400 fixes CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    wrapper = ROOT / PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH
    if (
        not wrapper.is_file()
        or wrapper.is_symlink()
        or file_sha256(wrapper)
        != PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256
    ):
        raise RuntimeError("fixed temperature wrapper changed")
    cmdline = Path("/proc") / str(os.getppid()) / "cmdline"
    tokens = tuple(
        value.decode("utf-8", errors="strict")
        for value in cmdline.read_bytes().split(b"\0")
        if value
    )
    candidates: list[Path] = []
    for token in tokens:
        if not token.endswith("run_with_gpu_temperature_control.py"):
            continue
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            candidates.append(candidate.resolve(strict=True))
        except OSError:
            pass
    if candidates != [wrapper.resolve(strict=True)]:
        raise RuntimeError("bounded-400 requires the fixed GPU wrapper")
    expected_flags = {
        "--gpu": PACRE_BOUNDED_VISIBLE_GPU,
        "--pause-temp": str(PACRE_BOUNDED_PAUSE_TEMPERATURE_C),
        "--resume-temp": str(PACRE_BOUNDED_RESUME_TEMPERATURE_C),
    }
    for name, expected in expected_flags.items():
        if _flag_value(tokens, name) != expected:
            raise RuntimeError(f"temperature wrapper {name} changed")
    return {
        "device": PACRE_BOUNDED_DEVICE,
        "CUDA_VISIBLE_DEVICES": PACRE_BOUNDED_VISIBLE_GPU,
        "CUBLAS_WORKSPACE_CONFIG": PACRE_BOUNDED_CUBLAS_WORKSPACE_CONFIG,
        "temperature_wrapper_repo_path": (
            PACRE_BOUNDED_TEMPERATURE_WRAPPER_REPO_PATH
        ),
        "temperature_wrapper_file_sha256": (
            PACRE_BOUNDED_TEMPERATURE_WRAPPER_FILE_SHA256
        ),
        "pause_temperature_c": PACRE_BOUNDED_PAUSE_TEMPERATURE_C,
        "resume_temperature_c": PACRE_BOUNDED_RESUME_TEMPERATURE_C,
    }


def _claim_output(attempt: Mapping[str, object]) -> Path:
    output = PACRE_BOUNDED_OUTPUT_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"single-use bounded output already exists: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / INCOMPLETE_FILE).open("xb").close()
    write_new_json(output / "attempt.json", attempt)
    (output / "receipts").mkdir(exist_ok=False)
    (output / "checkpoints").mkdir(exist_ok=False)
    return output


def _artifact_hashes(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name not in {
            INCOMPLETE_FILE,
            "COMPLETE.json",
            "FAILURE.json",
        }
    }


def _write_checkpoint(
    output: Path,
    model: torch.nn.Module,
) -> dict[str, object]:
    from safetensors.torch import load_file, save

    state = {
        name: value.detach().to("cpu").contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    raw = save(state)
    path = (
        output
        / "checkpoints"
        / f"{PACRE_PMOPE_OBJECTIVE}.safetensors"
    )
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    loaded = load_file(str(path), device="cpu")
    if set(loaded) != set(state) or any(
        not torch.equal(loaded[name], state[name]) for name in state
    ):
        raise RuntimeError("bounded checkpoint roundtrip changed")
    receipt = fingerprinted(
        {
            "schema_version": (
                "cure-lite-pacre-v23-vc-checkpoint-v1"
            ),
            "run_id": PACRE_BOUNDED_RUN_ID,
            "candidate": PACRE_VC_CANDIDATE,
            "objective": PACRE_PMOPE_OBJECTIVE,
            "repo_path": str(path.relative_to(ROOT)),
            "file_sha256": file_sha256(path),
            "state_keys": list(state),
            "serialization": "safetensors",
            "weights_only_roundtrip_verified": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    write_new_json(
        output
        / "checkpoints"
        / f"{PACRE_PMOPE_OBJECTIVE}.checkpoint.json",
        receipt,
    )
    return receipt


def run_once() -> dict[str, object]:
    """Execute and seal the sole bounded-400 attempt."""

    stabilize_pacre_vc_numerical_runtime()
    dr_verification = verify_dr_terminal()
    if (
        dr_verification.get("gate_passed") is not True
        or dr_verification.get("decision")
        != PACRE_VC_DR_PASS_DECISION
    ):
        raise PermissionError("bounded-400 requires terminal v23 D_R PASS")
    runtime = _runtime_contract()
    dataset_free = read_strict_json(
        protocol_root() / "dataset_free_receipt.json"
    )
    dataset_fingerprint = str(
        dataset_free["receipt_fingerprint"]
    )
    dr_wrapper = read_strict_json(
        Path(dr_verification["output"])
        / "receipts"
        / "dr_gate.json"
    ) if "output" in dr_verification else read_strict_json(
        ROOT
        / "runs/irstd1k_stage_a_seed42"
        / "pacre_v23_verifier_corrected_D_R_structural_r1"
        / "receipts/dr_gate.json"
    )
    dr_payload = dr_wrapper.get("receipt")
    if not isinstance(dr_payload, Mapping):
        raise TypeError("terminal D_R receipt is absent")
    dr_receipt = pacre_vc_dr_receipt_from_payload(dr_payload)

    model_config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=PACRE_FORMAL_FEATURE_CHANNELS,
        feature_stride=PACRE_FORMAL_FEATURE_STRIDE,
        width=PACRE_FORMAL_WIDTH,
    )
    config_body = {
        "schema_version": (
            "cure-lite-pacre-v23-vc-bounded-400-config-v1"
        ),
        "run_id": PACRE_BOUNDED_RUN_ID,
        "candidate": PACRE_VC_CANDIDATE,
        "model_config": {
            "feature_channels": PACRE_FORMAL_FEATURE_CHANNELS,
            "feature_stride": PACRE_FORMAL_FEATURE_STRIDE,
            "width": PACRE_FORMAL_WIDTH,
            "parameter_count": PACRE_FORMAL_PARAMETER_COUNT,
        },
        "budget": {
            "seed": COVERAGE_STATE_BOUNDED_SEED,
            "epochs": COVERAGE_STATE_BOUNDED_EPOCHS,
            "steps_per_epoch": (
                COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
            ),
            "updates": COVERAGE_STATE_BOUNDED_UPDATES,
        },
        "D_R_gate_receipt_fingerprint": (
            dr_receipt.receipt_fingerprint
        ),
        "dataset_free_receipt_fingerprint": dataset_fingerprint,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "formal_800_authorized": False,
    }
    config_fingerprint = stable_fingerprint(config_body)
    attempt = fingerprinted(
        {
            "schema_version": PACRE_BOUNDED_ATTEMPT_RECEIPT_SCHEMA,
            "run_id": PACRE_BOUNDED_RUN_ID,
            "output_repo_path": PACRE_BOUNDED_OUTPUT_REPO_PATH,
            "config_fingerprint": config_fingerprint,
            "runtime": runtime,
            "candidate": PACRE_VC_CANDIDATE,
            "objective": PACRE_PMOPE_OBJECTIVE,
            "budget": {
                "seed": COVERAGE_STATE_BOUNDED_SEED,
                "epochs": COVERAGE_STATE_BOUNDED_EPOCHS,
                "steps_per_epoch": (
                    COVERAGE_STATE_BOUNDED_STEPS_PER_EPOCH
                ),
                "updates": COVERAGE_STATE_BOUNDED_UPDATES,
            },
            "process_identity": pacre_bounded_process_identity(),
            "dataset_free_receipt_fingerprint": dataset_fingerprint,
            "dataset_free_invocations_before_claim": 1,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    output = _claim_output(attempt)
    receipts = output / "receipts"
    try:
        output_claim = load_pacre_bounded_output_claim()
        write_new_json(
            receipts / "config.json",
            fingerprinted(config_body),
        )
        source_paths = frozen_real_dr_source_paths()
        real_inputs = build_coverage_state_real_dr_inputs(**source_paths)
        population = build_coverage_state_bounded_population(
            real_inputs.scalar_cache,
            seed=COVERAGE_STATE_BOUNDED_SEED,
        )
        preflight = prepare_coverage_state_bounded_preflight(population)
        if not preflight.training_authorized:
            raise PermissionError("common bounded preflight did not pass")
        dr_receipt.verify_unchanged(
            dataset_free_receipt=dataset_free,
            real_inputs=real_inputs,
            bounded_population=population,
        )
        write_new_json(
            receipts / "inputs.json",
            fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-pacre-v23-vc-bounded-inputs-v1"
                    ),
                    "run_id": PACRE_BOUNDED_RUN_ID,
                    "real_inputs_fingerprint": (
                        real_inputs.build_fingerprint
                    ),
                    "population_fingerprint": (
                        population.population_fingerprint
                    ),
                    "cache_fingerprint": (
                        population.cache.cache_fingerprint
                    ),
                    "preflight_fingerprint": (
                        preflight.preflight_fingerprint
                    ),
                    "D_R_gate_receipt_fingerprint": (
                        dr_receipt.receipt_fingerprint
                    ),
                    "D_R_accessed": True,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                }
            ),
        )
        authorization = prepare_pacre_bounded_run_authorization(
            preflight,
            dataset_free,
            dr_receipt,
            real_inputs,
            model_config,
            output_claim=output_claim,
            run_id=PACRE_BOUNDED_RUN_ID,
        )
        write_new_json(
            receipts / "authorization.json",
            fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-pacre-v23-vc-bounded-"
                        "authorization-wrapper-v1"
                    ),
                    "run_id": PACRE_BOUNDED_RUN_ID,
                    "authorization": (
                        authorization.canonical_payload()
                    ),
                    "authorization_fingerprint": (
                        authorization.authorization_fingerprint
                    ),
                    "training_authorized": True,
                    "formal_800_authorized": False,
                }
            ),
        )
        result = run_pacre_pmope_bounded_400(
            authorization,
            model_config,
            run_id=PACRE_BOUNDED_RUN_ID,
            device=PACRE_BOUNDED_DEVICE,
        )
        checkpoint = _write_checkpoint(
            output,
            result.training.model,
        )
        write_new_json(
            receipts / "training.json",
            fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-pacre-v23-vc-bounded-training-v1"
                    ),
                    "run_id": PACRE_BOUNDED_RUN_ID,
                    "training_receipt": (
                        result.training.receipt.canonical_payload()
                    ),
                    "training_result": (
                        result.training.training_result.canonical_payload()
                    ),
                    "bundle_fingerprint": (
                        result.training.bundle_fingerprint
                    ),
                    "checkpoint_receipt_fingerprint": (
                        checkpoint["receipt_fingerprint"]
                    ),
                    "D_R_accessed": True,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                    "training_performed": True,
                }
            ),
        )
        write_new_json(
            receipts / "zero_level.json",
            fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-pacre-v23-vc-zero-level-v1"
                    ),
                    "run_id": PACRE_BOUNDED_RUN_ID,
                    "threshold": 0.0,
                    "threshold_search_performed": False,
                    "diagnostic": result.diagnostic.canonical_payload(),
                    "diagnostic_fingerprint": (
                        result.diagnostic.result_fingerprint
                    ),
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                }
            ),
        )
        write_new_json(
            receipts / "bounded_result.json",
            fingerprinted(
                {
                    "schema_version": (
                        "cure-lite-pacre-v23-vc-bounded-result-wrapper-v1"
                    ),
                    "run_id": PACRE_BOUNDED_RUN_ID,
                    "result": result.canonical_payload(),
                    "result_fingerprint": result.result_fingerprint,
                }
            ),
        )
        decision = fingerprinted(
            {
                "schema_version": (
                    "cure-lite-pacre-v23-vc-bounded-decision-v1"
                ),
                "run_id": PACRE_BOUNDED_RUN_ID,
                "status": (
                    "PACRE_V23_VC_BOUNDED_400_GATE_PASS"
                    if result.bounded_gate_passed
                    else "PACRE_V23_VC_BOUNDED_400_GATE_FAIL"
                ),
                "bounded_gate_passed": result.bounded_gate_passed,
                "failed_checks": list(result.failed_checks),
                "result_fingerprint": result.result_fingerprint,
                "formal800_eligible": result.formal800_eligible,
                "formal_800_authorized": False,
                "formal_800_executed": False,
                "D_R_accessed": True,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "training_performed": True,
            }
        )
        write_new_json(receipts / "decision.json", decision)
        hashes = _artifact_hashes(output)
        complete_body: dict[str, object] = {
            "schema_version": (
                "cure-lite-pacre-v23-vc-bounded-complete-v1"
            ),
            "run_id": PACRE_BOUNDED_RUN_ID,
            "status": decision["status"],
            "bounded_gate_passed": result.bounded_gate_passed,
            "failed_checks": list(result.failed_checks),
            "result_fingerprint": result.result_fingerprint,
            "artifact_files": hashes,
            "artifact_count": len(hashes),
            "completed_updates": COVERAGE_STATE_BOUNDED_UPDATES,
            "formal800_eligible": result.formal800_eligible,
            "formal_800_authorized": False,
            "D_R_accessed": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": True,
        }
        complete = {
            **complete_body,
            "complete_fingerprint": stable_fingerprint(complete_body),
        }
        write_new_json(output / "COMPLETE.json", complete)
        (output / INCOMPLETE_FILE).unlink()
        return {
            "run_id": PACRE_BOUNDED_RUN_ID,
            "output": str(output),
            "status": decision["status"],
            "bounded_gate_passed": result.bounded_gate_passed,
            "failed_checks": list(result.failed_checks),
            "formal800_eligible": result.formal800_eligible,
            "formal_800_authorized": False,
            "complete_fingerprint": complete["complete_fingerprint"],
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    except BaseException as error:
        failure = fingerprinted(
            {
                "schema_version": (
                    "cure-lite-pacre-v23-vc-bounded-failure-v1"
                ),
                "run_id": PACRE_BOUNDED_RUN_ID,
                "exception_type": type(error).__name__,
                "message": str(error),
                "artifact_files_before_failure": (
                    _artifact_hashes(output)
                ),
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        try:
            write_new_json(output / "FAILURE.json", failure)
        except BaseException:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-once", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    result = run_once()
    import json

    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if result["bounded_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
