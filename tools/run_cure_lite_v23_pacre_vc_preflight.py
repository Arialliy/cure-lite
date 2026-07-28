#!/usr/bin/env python3
"""Create or validate PACRE-VC generated-only evidence and D_R authorization.

``--run-generated`` creates every source/runtime/parity/stress/dataset-free
receipt without opening new ``D_R`` tensors.  ``--authorize-dr`` is a separate
final metadata step and creates the sole pre-run authorization.  There is no
``D_V`` or ``D_T`` option.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite_v23.authorization import (
    build_dr_pre_run_authorization,
    build_runner_verification_receipt,
    build_v22_failure_inheritance_receipt,
    dr_output_path,
    protocol_root,
    verify_dr_pre_run_authorization,
)
from cure_lite_v23.dataset_free import (
    run_pacre_vc_dataset_free_gate,
)
from cure_lite_v23.environment import (
    fingerprinted_runtime_environment,
    stabilize_pacre_vc_numerical_runtime,
)
from cure_lite_v23.numeric_stress import (
    run_pacre_vc_formal_numeric_stress_receipt,
    run_pacre_vc_scalar_counterexample_receipt,
)
from cure_lite_v23.parity import (
    run_pacre_vc_generated_parity_receipt,
)
from cure_lite_v23.protocol import (
    read_strict_json,
    source_closure_payload,
    write_new_json,
)


GENERATED_FILES = {
    "v22_failure_inheritance": "v22_failure_inheritance_receipt.json",
    "runtime_cpu": "runtime_environment_cpu_lock.json",
    "runtime_selected_device": "runtime_environment_lock.json",
    "implementation_closure": "implementation_closure.json",
    "forward_parity": "forward_parity_receipt.json",
    "scalar_counterexample": "scalar_counterexample_receipt.json",
    "cpu_stress": "formal_shape_cpu_stress_receipt.json",
    "selected_device_stress": (
        "formal_shape_selected_device_stress_receipt.json"
    ),
    "dataset_free": "dataset_free_receipt.json",
    "runner_verification": "runner_verification_receipt.json",
}
AUTHORIZATION_FILE = "D_R_pre_run_authorization.json"


def _paths() -> dict[str, Path]:
    root = protocol_root()
    return {
        name: root / filename
        for name, filename in GENERATED_FILES.items()
    }


def _read_generated() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, path in _paths().items():
        result[name] = read_strict_json(path)
    return result


def _require_clean_generated_targets() -> None:
    root = protocol_root()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(
            "protocol root/design preregistration must exist first"
        )
    authorization = root / AUTHORIZATION_FILE
    if authorization.exists() or authorization.is_symlink():
        raise FileExistsError("D_R authorization already exists")
    existing = [
        str(path)
        for path in _paths().values()
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "generated preflight artifacts already exist: "
            + ", ".join(existing)
        )
    if dr_output_path().exists() or dr_output_path().is_symlink():
        raise FileExistsError("D_R result directory already exists")


def run_generated() -> dict[str, object]:
    """Create the complete generated-only prerequisite matrix once."""

    stabilize_pacre_vc_numerical_runtime()
    _require_clean_generated_targets()
    paths = _paths()
    source = source_closure_payload()
    runtime_cpu = fingerprinted_runtime_environment("cpu")
    runtime_cuda = fingerprinted_runtime_environment("cuda:0")

    # Every generated computation receives the already-frozen live locks.
    parity = run_pacre_vc_generated_parity_receipt(include_cuda=True)
    counterexample = run_pacre_vc_scalar_counterexample_receipt()
    cpu_stress = run_pacre_vc_formal_numeric_stress_receipt(
        device="cpu",
        runtime_environment=runtime_cpu,
        source_closure=source,
    )
    cuda_stress = run_pacre_vc_formal_numeric_stress_receipt(
        device="cuda:0",
        runtime_environment=runtime_cuda,
        source_closure=source,
    )
    dataset_free = run_pacre_vc_dataset_free_gate(
        parity_receipt=parity,
        cpu_stress_receipt=cpu_stress,
        selected_device_stress_receipt=cuda_stress,
        counterexample_receipt=counterexample,
        runtime_environment_receipts={
            "cpu": runtime_cpu,
            "cuda:0": runtime_cuda,
        },
        source_closure_receipt=source,
    )
    v22_failure = build_v22_failure_inheritance_receipt()
    runner = build_runner_verification_receipt()
    payloads: dict[str, Mapping[str, object]] = {
        "v22_failure_inheritance": v22_failure,
        "runtime_cpu": runtime_cpu,
        "runtime_selected_device": runtime_cuda,
        "implementation_closure": source,
        "forward_parity": parity,
        "scalar_counterexample": counterexample,
        "cpu_stress": cpu_stress,
        "selected_device_stress": cuda_stress,
        "dataset_free": dataset_free,
        "runner_verification": runner,
    }
    for name in GENERATED_FILES:
        write_new_json(paths[name], payloads[name])
    gate_passed = bool(
        parity.get("gate_passed")
        and counterexample.get("gate_passed")
        and cpu_stress.get("gate_passed")
        and cuda_stress.get("gate_passed")
        and dataset_free.get("gate_passed")
    )
    return {
        "mode": "generated_only_preflight",
        "protocol_root": str(protocol_root()),
        "artifact_count": len(payloads),
        "gate_passed": gate_passed,
        "authorization_created": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }


def authorize_dr() -> dict[str, object]:
    """Create the final pre-run authorization after all gates pass."""

    stabilize_pacre_vc_numerical_runtime()
    path = protocol_root() / AUTHORIZATION_FILE
    if path.exists() or path.is_symlink():
        raise FileExistsError("D_R authorization already exists")
    generated = _read_generated()
    authorization = build_dr_pre_run_authorization(
        **generated,
    )
    write_new_json(path, authorization)
    return {
        "mode": "authorize_D_R",
        "authorization": str(path),
        "authorization_fingerprint": (
            authorization["authorization_fingerprint"]
        ),
        "status": authorization["status"],
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }


def validate_generated() -> dict[str, object]:
    """Independently rebuild all generated bindings without writing."""

    stabilize_pacre_vc_numerical_runtime()
    generated = _read_generated()
    path = protocol_root() / AUTHORIZATION_FILE
    output_exists = (
        dr_output_path().exists() or dr_output_path().is_symlink()
    )
    authorization_exists = path.is_file() and not path.is_symlink()
    if authorization_exists:
        authorization = read_strict_json(path)
        authorization_fingerprint = (
            verify_dr_pre_run_authorization(
                authorization,
                require_output_absent=not output_exists,
                **generated,
            )
        )
    else:
        if output_exists:
            raise RuntimeError(
                "D_R output exists without a canonical authorization"
            )
        expected = build_dr_pre_run_authorization(**generated)
        authorization_fingerprint = expected[
            "authorization_fingerprint"
        ]
    return {
        "mode": "validate_generated",
        "generated_artifact_count": len(generated),
        "authorization_exists": authorization_exists,
        "authorization_binding_fingerprint": (
            authorization_fingerprint
        ),
        "all_generated_bindings_valid": True,
        "D_R_output_exists": output_exists,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-generated", action="store_true")
    mode.add_argument("--authorize-dr", action="store_true")
    mode.add_argument("--validate-generated", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_generated:
        result = run_generated()
    elif args.authorize_dr:
        result = authorize_dr()
    else:
        result = validate_generated()
    import json

    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("gate_passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
