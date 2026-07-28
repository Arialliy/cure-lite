#!/usr/bin/env python3
"""Run the single read-only v20 BFA curvature audit on frozen real ``D_R``.

The command is deliberately narrower than a training entrypoint.  It binds
the completed v20-r2 run byte-for-byte, reconstructs only its frozen ``D_R``
population, loads the exact tensor-only BFA checkpoint, and calls the
curvature audit once under ``torch.inference_mode``.  It exposes no seed,
checkpoint, split, output, retry, resume, threshold, or parameter option.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from cure_lite.experiment.coverage_state_bfa_dataset_free import (
    COVERAGE_STATE_BFA_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_BFA_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_BFA_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_BFA_FORMAL_WIDTH,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    build_coverage_state_real_dr_inputs,
)
from cure_lite.frozen_base import module_state_fingerprint
from tools import (
    run_coverage_state_bfa_cmif_pmope_bounded_400 as _v20_cli,
)


_ROOT = Path(__file__).resolve().parents[1]

SOURCE_RUN_ID = "cure_lite_bfa_cmif_v20_pmope_bounded_400_r2"
SOURCE_RUN_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{SOURCE_RUN_ID}"
SOURCE_RUN_PATH = _ROOT / SOURCE_RUN_REPO_PATH
SOURCE_COMPLETE_RELATIVE = "COMPLETE.json"
SOURCE_ZERO_RELATIVE = "receipts/zero_level.json"
SOURCE_CHECKPOINT_RELATIVE = "checkpoints/pmope_joint.safetensors"
SOURCE_CHECKPOINT_RECEIPT_RELATIVE = (
    "checkpoints/pmope_joint.checkpoint.json"
)
SOURCE_INPUT_RECEIPT_RELATIVE = "receipts/inputs.json"

SOURCE_COMPLETE_SHA256 = (
    "a1307929615ef877726387df024c90de33b54924ea2c22de2c7f7f5a51e7f334"
)
SOURCE_COMPLETE_FINGERPRINT = (
    "8908a8c1896951e46fd737aa6f7fef2c9935e6524632b3576b8069faa026e2eb"
)
SOURCE_ZERO_SHA256 = (
    "fe9820d72fc796aa0e70045c48d3b40eb05e0aba2bb5a3447bff3830e9cfadc4"
)
SOURCE_ZERO_RECEIPT_FINGERPRINT = (
    "4301c8e9f3393c2bc64c28b20e3b6e16bdc98b974281b7d9c67d239a86c76219"
)
SOURCE_ZERO_DIAGNOSTIC_FINGERPRINT = (
    "50a92452a04d2a40f735c4e4cef75ce50df4ecf98338ce145f344bd6a76b3b77"
)
SOURCE_CHECKPOINT_SHA256 = (
    "040d2ca4ffa012c813e2c3e5dfa2c6f4877a91c8ff0b901bf8dc83df62026c42"
)
SOURCE_CHECKPOINT_RECEIPT_SHA256 = (
    "886d1923641825c4650b8e867e62475c5c7d9447b61973079a19e0856402aae1"
)
SOURCE_CHECKPOINT_RECEIPT_FINGERPRINT = (
    "e6edd17860f2250eb9e96d045424d0d73ef13e0a5152309f34b1655998f007ff"
)
SOURCE_MODULE_STATE_FINGERPRINT = (
    "0393532f8ea62e790c120ca0c0b86bf04c67b88c863e333f3f7c640d865ab5c0"
)
SOURCE_INPUT_RECEIPT_SHA256 = (
    "40b12f4c6c943ece8d0e62d7c3e15efafd4c3d66dc0b8ebb3a7f248081b9e0a4"
)
SOURCE_INPUT_RECEIPT_FINGERPRINT = (
    "65c9319f46142c31f32b313b7787193741b0afc3af37a7799ca0c1e8cd6a5252"
)
SOURCE_POPULATION_FINGERPRINT = (
    "1a53467d57bea595afcc1edd3330708d1dda39e0e2d606325e552e8993e7841c"
)
SOURCE_CACHE_FINGERPRINT = (
    "c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca00754c206092aaf66d8"
)

RUN_ID = "cure_lite_bfa_cmif_v20_curvature_audit_r1"
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = _ROOT / OUTPUT_REPO_PATH
RUN_SCHEMA = "cure-lite-bfa-v20-curvature-audit-run-v1"
RECEIPT_SCHEMA = "cure-lite-bfa-v20-curvature-audit-receipt-v1"
VALIDATION_SCHEMA = (
    "cure-lite-bfa-v20-curvature-audit-binding-validation-v1"
)
FROZEN_DEVICE = "cuda:0"
FROZEN_SEED = 42
FROZEN_FEATURE_CHANNELS = COVERAGE_STATE_BFA_FORMAL_FEATURE_CHANNELS
FROZEN_FEATURE_STRIDE = COVERAGE_STATE_BFA_FORMAL_FEATURE_STRIDE
FROZEN_MODEL_WIDTH = COVERAGE_STATE_BFA_FORMAL_WIDTH
FROZEN_PARAMETER_COUNT = COVERAGE_STATE_BFA_FORMAL_PARAMETER_COUNT
_INCOMPLETE = ".incomplete"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fingerprinted(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    if "receipt_fingerprint" in result:
        raise ValueError("payload already contains receipt_fingerprint")
    result["receipt_fingerprint"] = stable_fingerprint(result)
    return result


def _strict_regular_file(path: Path, *, name: str) -> Path:
    absolute = Path(os.path.abspath(path))
    if (
        path.is_symlink()
        or path.resolve(strict=True) != absolute
        or not absolute.is_file()
    ):
        raise RuntimeError(f"{name} path changed")
    return absolute


def _load_json_object(path: Path, *, name: str) -> dict[str, object]:
    absolute = _strict_regular_file(path, name=name)
    try:
        payload = json.loads(absolute.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is not canonical JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain one JSON object")
    return payload


def _verify_self_fingerprint(
    payload: Mapping[str, object],
    *,
    field: str,
    expected: str,
    name: str,
) -> None:
    observed = payload.get(field)
    if observed != expected or not _is_sha256(observed):
        raise RuntimeError(f"{name} fingerprint binding changed")
    body = dict(payload)
    body.pop(field, None)
    if stable_fingerprint(body) != expected:
        raise RuntimeError(f"{name} self-fingerprint is inconsistent")


def _verify_frozen_v20_bindings() -> dict[str, object]:
    """Verify the exact completed v20-r2 authority without modifying it."""

    complete_path = SOURCE_RUN_PATH / SOURCE_COMPLETE_RELATIVE
    zero_path = SOURCE_RUN_PATH / SOURCE_ZERO_RELATIVE
    checkpoint_path = SOURCE_RUN_PATH / SOURCE_CHECKPOINT_RELATIVE
    checkpoint_receipt_path = (
        SOURCE_RUN_PATH / SOURCE_CHECKPOINT_RECEIPT_RELATIVE
    )
    input_receipt_path = (
        SOURCE_RUN_PATH / SOURCE_INPUT_RECEIPT_RELATIVE
    )
    expected_hashes = {
        "complete": (complete_path, SOURCE_COMPLETE_SHA256),
        "zero": (zero_path, SOURCE_ZERO_SHA256),
        "checkpoint": (checkpoint_path, SOURCE_CHECKPOINT_SHA256),
        "checkpoint_receipt": (
            checkpoint_receipt_path,
            SOURCE_CHECKPOINT_RECEIPT_SHA256,
        ),
        "input_receipt": (
            input_receipt_path,
            SOURCE_INPUT_RECEIPT_SHA256,
        ),
    }
    for name, (path, expected) in expected_hashes.items():
        _strict_regular_file(path, name=f"v20 {name}")
        if file_sha256(path) != expected:
            raise RuntimeError(f"v20 {name} bytes changed")

    complete = _load_json_object(complete_path, name="v20 COMPLETE")
    zero = _load_json_object(zero_path, name="v20 zero receipt")
    checkpoint_receipt = _load_json_object(
        checkpoint_receipt_path,
        name="v20 checkpoint receipt",
    )
    input_receipt = _load_json_object(
        input_receipt_path,
        name="v20 input receipt",
    )
    complete_body = dict(complete)
    complete_body.pop("complete_fingerprint", None)
    if (
        complete.get("complete_fingerprint")
        != SOURCE_COMPLETE_FINGERPRINT
        or stable_fingerprint(complete_body)
        != SOURCE_COMPLETE_FINGERPRINT
    ):
        raise RuntimeError("v20 COMPLETE self-fingerprint is inconsistent")
    _verify_self_fingerprint(
        zero,
        field="receipt_fingerprint",
        expected=SOURCE_ZERO_RECEIPT_FINGERPRINT,
        name="v20 zero receipt",
    )
    _verify_self_fingerprint(
        checkpoint_receipt,
        field="receipt_fingerprint",
        expected=SOURCE_CHECKPOINT_RECEIPT_FINGERPRINT,
        name="v20 checkpoint receipt",
    )
    _verify_self_fingerprint(
        input_receipt,
        field="receipt_fingerprint",
        expected=SOURCE_INPUT_RECEIPT_FINGERPRINT,
        name="v20 input receipt",
    )
    artifact_files = complete.get("artifact_files")
    candidate_diagnostic = zero.get("candidate_diagnostic")
    bounded_population = input_receipt.get("bounded_population")
    if (
        complete.get("complete_fingerprint")
        != SOURCE_COMPLETE_FINGERPRINT
        or complete.get("status") != "complete"
        or complete.get("run_id") != SOURCE_RUN_ID
        or complete.get("runtime_splits") != ["D_R"]
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("training_invocations") != 1
        or not isinstance(artifact_files, Mapping)
        or artifact_files.get(SOURCE_ZERO_RELATIVE)
        != SOURCE_ZERO_SHA256
        or artifact_files.get(SOURCE_CHECKPOINT_RELATIVE)
        != SOURCE_CHECKPOINT_SHA256
        or artifact_files.get(SOURCE_CHECKPOINT_RECEIPT_RELATIVE)
        != SOURCE_CHECKPOINT_RECEIPT_SHA256
        or artifact_files.get(SOURCE_INPUT_RECEIPT_RELATIVE)
        != SOURCE_INPUT_RECEIPT_SHA256
        or zero.get("diagnostic_result_fingerprint")
        != SOURCE_ZERO_DIAGNOSTIC_FINGERPRINT
        or not isinstance(candidate_diagnostic, Mapping)
        or candidate_diagnostic.get("checkpoint_fingerprint")
        != SOURCE_MODULE_STATE_FINGERPRINT
        or checkpoint_receipt.get("checkpoint_file_sha256")
        != SOURCE_CHECKPOINT_SHA256
        or checkpoint_receipt.get("module_state_fingerprint")
        != SOURCE_MODULE_STATE_FINGERPRINT
        or checkpoint_receipt.get("model_class")
        != "CURELiteBinaryFlipAntisymmetricLevelSet"
        or checkpoint_receipt.get("objective") != "pmope_joint"
        or checkpoint_receipt.get("serialization") != "safetensors"
        or checkpoint_receipt.get("tensor_only_state_dict") is not True
        or checkpoint_receipt.get("weights_only_roundtrip_verified")
        is not True
        or input_receipt.get("population_fingerprint")
        != SOURCE_POPULATION_FINGERPRINT
        or not isinstance(bounded_population, Mapping)
        or bounded_population.get("bounded_cache_fingerprint")
        != SOURCE_CACHE_FINGERPRINT
        or input_receipt.get("D_V_accessed") is not False
        or input_receipt.get("D_T_accessed") is not False
    ):
        raise RuntimeError("frozen v20 authority graph changed")
    return {
        "complete": complete,
        "zero": zero,
        "checkpoint_receipt": checkpoint_receipt,
        "input_receipt": input_receipt,
        "candidate_diagnostic": dict(candidate_diagnostic),
        "checkpoint_path": checkpoint_path,
    }


def _load_exact_bfa_model(
    checkpoint_path: Path,
    *,
    device: str,
) -> CURELiteBinaryFlipAntisymmetricLevelSet:
    """Load the exact frozen BFA state without constructing training state."""

    if device != FROZEN_DEVICE:
        raise ValueError("curvature audit fixes device to cuda:0")
    if file_sha256(checkpoint_path) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("v20 checkpoint bytes changed before load")
    config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=FROZEN_FEATURE_CHANNELS,
        feature_stride=FROZEN_FEATURE_STRIDE,
        width=FROZEN_MODEL_WIDTH,
    )
    model = CURELiteBinaryFlipAntisymmetricLevelSet(config)
    if sum(value.numel() for value in model.parameters()) != (
        FROZEN_PARAMETER_COUNT
    ):
        raise RuntimeError("exact BFA parameter count changed")
    from safetensors.torch import load_file

    state = load_file(str(checkpoint_path), device="cpu")
    expected_keys = set(model.state_dict())
    if set(state) != expected_keys:
        raise RuntimeError("v20 checkpoint state keys changed")
    model.load_state_dict(state, strict=True)
    if module_state_fingerprint(model) != SOURCE_MODULE_STATE_FINGERPRINT:
        raise RuntimeError("loaded BFA module state changed")
    model.requires_grad_(False)
    model.eval()
    return model.to(device=device)


def _build_frozen_population() -> tuple[object, object]:
    """Rebuild only the hash-bound v20 real-``D_R`` population."""

    source_paths = _v20_cli._verify_frozen_sources()
    real_inputs = build_coverage_state_real_dr_inputs(**source_paths)
    population = build_coverage_state_bounded_population(
        real_inputs.scalar_cache,
        seed=FROZEN_SEED,
    )
    if (
        getattr(population, "population_fingerprint", None)
        != SOURCE_POPULATION_FINGERPRINT
        or getattr(population.cache, "cache_fingerprint", None)
        != SOURCE_CACHE_FINGERPRINT
    ):
        raise RuntimeError("reconstructed v20 D_R population changed")
    return real_inputs, population


def _audit_payload(value: object) -> dict[str, object]:
    if hasattr(value, "verify"):
        value.verify()  # type: ignore[attr-defined]
    if hasattr(value, "canonical_payload"):
        payload = value.canonical_payload()  # type: ignore[attr-defined]
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("curvature audit must return a receipt-like object")
    if not isinstance(payload, Mapping):
        raise TypeError("curvature audit canonical payload must be a mapping")
    result = dict(payload)
    execution = result.get("execution")
    if (
        not isinstance(result.get("decision"), str)
        or not isinstance(execution, Mapping)
        or execution.get("training_performed") is not False
        or execution.get("backward_performed") is not False
        or execution.get("optimizer_constructed") is not False
        or execution.get("D_V_accessed") is not False
        or execution.get("D_T_accessed") is not False
    ):
        raise RuntimeError("curvature audit read-only evidence changed")
    claimed = getattr(value, "receipt_fingerprint", None)
    if claimed is not None:
        if not _is_sha256(claimed) or stable_fingerprint(result) != claimed:
            raise RuntimeError("curvature audit fingerprint changed")
    return result


def _implementation_binding() -> dict[str, str]:
    relatives = (
        "tools/run_coverage_state_bfa_curvature_audit.py",
        "cure_lite/experiment/coverage_state_bfa_curvature_audit.py",
        "cure_lite/coverage_state_binary_flip_antisymmetric.py",
    )
    return {
        relative: file_sha256(
            _strict_regular_file(
                _ROOT / relative,
                name=f"curvature implementation {relative}",
            )
        )
        for relative in relatives
    }


def validate_bindings() -> dict[str, object]:
    """Validate frozen sources without loading tensor data or claiming output."""

    frozen = _verify_frozen_v20_bindings()
    implementation = _implementation_binding()
    return _fingerprinted(
        {
            "schema_version": VALIDATION_SCHEMA,
            "status": "bindings_valid",
            "source_run_id": SOURCE_RUN_ID,
            "source_complete_fingerprint": (
                frozen["complete"]["complete_fingerprint"]  # type: ignore[index]
            ),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_zero_receipt_fingerprint": (
                SOURCE_ZERO_RECEIPT_FINGERPRINT
            ),
            "implementation": implementation,
            "implementation_fingerprint": stable_fingerprint(
                implementation
            ),
            "output_exists": (
                OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink()
            ),
            "D_R_tensor_payload_accessed": False,
            "runtime_splits": [],
            "training_performed": False,
            "backward_performed": False,
            "optimizer_constructed": False,
            "optimizer_step_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "threshold_search_performed": False,
            "parameter_search_performed": False,
            "output_claimed": False,
            "not_a_scientific_result": True,
        }
    )


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _claim_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"single-use curvature audit output already exists: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    receipts = output / "receipts"
    receipts.mkdir()
    with (output / _INCOMPLETE).open("xb") as handle:
        handle.write(b"INCOMPLETE\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipts


def _run_audit(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
    real_inputs: object,
    population: object,
    baseline_diagnostic: Mapping[str, object],
    *,
    device: str,
) -> object:
    # Delayed import keeps ``--validate-bindings`` free of real-audit module
    # initialization and isolates this entrypoint from interface evolution.
    from cure_lite.experiment import coverage_state_bfa_curvature_audit as audit

    preferred = getattr(audit, "run_bfa_curvature_audit", None)
    if preferred is not None:
        return preferred(
            model,
            population,
            baseline_diagnostic,
            device=device,
        )
    # The current core exposes a stricter frozen-r2 wrapper.  Preserve the
    # requested baseline-diagnostic argument as a validated binding, then
    # delegate to that wrapper without weakening any frozen evidence edge.
    if (
        stable_fingerprint(dict(baseline_diagnostic))
        != SOURCE_ZERO_DIAGNOSTIC_FINGERPRINT
    ):
        raise RuntimeError("baseline diagnostic binding changed")
    return audit.audit_frozen_coverage_state_bfa_v20_r2_curvature_checkpoint(
        model,
        real_inputs,
        population,
        device=device,
        complete_fingerprint=SOURCE_COMPLETE_FINGERPRINT,
        zero_receipt_fingerprint=SOURCE_ZERO_RECEIPT_FINGERPRINT,
        diagnostic_fingerprint=SOURCE_ZERO_DIAGNOSTIC_FINGERPRINT,
        checkpoint_file_sha256=SOURCE_CHECKPOINT_SHA256,
    )


def run_once() -> dict[str, object]:
    """Execute the sole frozen-checkpoint, real-``D_R`` read-only audit."""

    frozen = _verify_frozen_v20_bindings()
    implementation = _implementation_binding()
    receipts = _claim_output(OUTPUT_PATH)
    try:
        real_inputs, population = _build_frozen_population()
        model = _load_exact_bfa_model(
            frozen["checkpoint_path"],  # type: ignore[arg-type]
            device=FROZEN_DEVICE,
        )
        initial_state_fingerprint = module_state_fingerprint(model)
        with torch.inference_mode():
            audit = _run_audit(
                model,
                real_inputs,
                population,
                frozen["candidate_diagnostic"],  # type: ignore[arg-type]
                device=FROZEN_DEVICE,
            )
        if (
            model.training
            or any(parameter.requires_grad for parameter in model.parameters())
            or module_state_fingerprint(model) != initial_state_fingerprint
            or initial_state_fingerprint != SOURCE_MODULE_STATE_FINGERPRINT
        ):
            raise RuntimeError("read-only curvature audit changed model state")
        audit_body = _audit_payload(audit)
        receipt = _fingerprinted(
            {
                "schema_version": RECEIPT_SCHEMA,
                "status": "complete_read_only_curvature_audit",
                "source_run": {
                    "run_id": SOURCE_RUN_ID,
                    "complete_sha256": SOURCE_COMPLETE_SHA256,
                    "complete_fingerprint": (
                        SOURCE_COMPLETE_FINGERPRINT
                    ),
                    "zero_sha256": SOURCE_ZERO_SHA256,
                    "zero_receipt_fingerprint": (
                        SOURCE_ZERO_RECEIPT_FINGERPRINT
                    ),
                    "zero_diagnostic_fingerprint": (
                        SOURCE_ZERO_DIAGNOSTIC_FINGERPRINT
                    ),
                    "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                    "checkpoint_receipt_sha256": (
                        SOURCE_CHECKPOINT_RECEIPT_SHA256
                    ),
                    "checkpoint_receipt_fingerprint": (
                        SOURCE_CHECKPOINT_RECEIPT_FINGERPRINT
                    ),
                    "module_state_fingerprint": (
                        SOURCE_MODULE_STATE_FINGERPRINT
                    ),
                    "input_receipt_sha256": (
                        SOURCE_INPUT_RECEIPT_SHA256
                    ),
                    "input_receipt_fingerprint": (
                        SOURCE_INPUT_RECEIPT_FINGERPRINT
                    ),
                    "read_only": True,
                },
                "split": "D_R",
                "runtime_splits": ["D_R"],
                "population_fingerprint": (
                    SOURCE_POPULATION_FINGERPRINT
                ),
                "cache_fingerprint": SOURCE_CACHE_FINGERPRINT,
                "model_class": (
                    "CURELiteBinaryFlipAntisymmetricLevelSet"
                ),
                "model_parameter_count": FROZEN_PARAMETER_COUNT,
                "model_loaded_from_frozen_checkpoint": True,
                "model_state_unchanged": True,
                "baseline_diagnostic_source": (
                    "frozen_v20_r2_zero_level_receipt"
                ),
                "audit": audit_body,
                "audit_fingerprint": stable_fingerprint(audit_body),
                "decision": audit_body["decision"],
                "implementation": implementation,
                "implementation_fingerprint": stable_fingerprint(
                    implementation
                ),
                "execution": {
                    "device": FROZEN_DEVICE,
                    "inference_mode": True,
                    "audit_invocations": 1,
                    "population_construction_invocations": 1,
                    "checkpoint_load_invocations": 1,
                    "training_performed": False,
                    "backward_performed": False,
                    "optimizer_constructed": False,
                    "optimizer_step_performed": False,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                    "threshold_search_performed": False,
                    "parameter_search_performed": False,
                    "checkpoint_written": False,
                },
                "evidence_scope": {
                    "frozen_v20_checkpoint_diagnostic_only": True,
                    "new_model_result": False,
                    "bounded_400_result": False,
                    "formal_800_result": False,
                    "performance_claim_supported": False,
                    "formal_800_authorized": False,
                    "full_CURE_authorized": False,
                    "cross_backbone_authorized": False,
                },
            }
        )
        receipt_path = receipts / "curvature_audit.json"
        _write_new_json(receipt_path, receipt)
        complete = {
            "schema_version": RUN_SCHEMA,
            "status": "complete",
            "run_id": RUN_ID,
            "output_repo_path": OUTPUT_REPO_PATH,
            "source_run_id": SOURCE_RUN_ID,
            "source_complete_fingerprint": (
                SOURCE_COMPLETE_FINGERPRINT
            ),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "receipt_fingerprint": receipt["receipt_fingerprint"],
            "audit_fingerprint": receipt["audit_fingerprint"],
            "decision": receipt["decision"],
            "artifact_files": {
                "receipts/curvature_audit.json": file_sha256(
                    receipt_path
                )
            },
            "artifact_file_count": 1,
            "audit_invocations": 1,
            "training_performed": False,
            "backward_performed": False,
            "optimizer_constructed": False,
            "optimizer_step_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "threshold_search_performed": False,
            "parameter_search_performed": False,
            "checkpoint_written": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_claim_supported": False,
        }
        complete["complete_fingerprint"] = stable_fingerprint(complete)
        _write_new_json(OUTPUT_PATH / "COMPLETE.json", complete)
        (OUTPUT_PATH / _INCOMPLETE).unlink()
        return complete
    except BaseException:
        # Preserve the claimed directory and marker as an immutable failed
        # attempt.  No resume or automatic retry path is provided.
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen v20 BFA read-only curvature audit."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-bindings",
        action="store_true",
        help="validate frozen byte bindings without reading D_R tensors",
    )
    mode.add_argument(
        "--run-once",
        action="store_true",
        help="consume the sole read-only D_R curvature audit attempt",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_bindings() if args.validate_bindings else run_once()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
