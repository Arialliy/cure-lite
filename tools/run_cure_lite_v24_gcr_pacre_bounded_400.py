#!/usr/bin/env python3
"""Run or verify the exact v24 paired bounded-400 stage.

The input factory must return an already prepared
``GCRPACREBoundedAuthorization`` whose protocol tokens were reissued in this
process.  This CLI exposes no epoch, step, seed, retry, resume, D_V, or D_T
override.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite_v24.artifact_io import (
    atomic_write_new_canonical_json,
    read_canonical_json,
)
from cure_lite_v24.bounded_runner import (
    GCRPACREBoundedAuthorization,
    build_paired_bounded_receipt,
    run_gcr_pacre_paired_bounded_400,
)
from cure_lite_v24.bounded_run_start import (
    VerifiedGCRPACREBoundedChainConfig,
    create_gcr_pacre_bounded_run_start_marker,
    load_and_verify_gcr_pacre_bounded_chain_config,
)
from cure_lite_v24.fixed_dr_evaluator import FrozenGCRPACREDREvaluator
from cure_lite_v24.source_closure import (
    assert_gcr_pacre_v24_loaded_source_closure_complete,
    audit_gcr_pacre_v24_loaded_source_closure,
)
from tools.gcr_pacre_v24_protocol import (
    decide_paired_bounded400,
    validate_paired_bounded_receipt,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_INPUT_FACTORY = (
    "cure_lite_v24.real_input_factory:"
    "build_gcr_pacre_v24_stage_authorization"
)
AUTHORIZATION_FILE = "bounded_400_authorization.json"
RESULT_FILE = "bounded_400_result.json"
DECISION_FILE = "bounded_400_decision.json"
DIAGNOSTICS_FILE = "bounded_400_diagnostics.json"


def _preflight_execution_device(value: object) -> str:
    """Validate and initialize one exact CPU or explicitly indexed CUDA device."""

    if type(value) is not str or not value:
        raise TypeError("device must be a non-empty string")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid torch device: {value!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device type must be cpu or cuda")
    if device.type == "cuda":
        prefix, separator, index_text = value.partition(":")
        if (
            prefix != "cuda"
            or not separator
            or not index_text.isdecimal()
            or device.index is None
        ):
            raise ValueError("CUDA device index must be explicit")
        requested_index = int(index_text)
        if requested_index != device.index:
            raise ValueError("CUDA device index is out of range")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device_count = torch.cuda.device_count()
        if requested_index >= device_count:
            raise ValueError(
                "CUDA device index is out of range: "
                f"{requested_index} >= {device_count}"
            )
    try:
        torch.empty(1, device=device)
    except Exception as exc:
        raise RuntimeError(
            f"failed to initialize execution device {device}"
        ) from exc
    return str(device)


def _load_authorization(
    specification: str,
    *,
    chain_config: VerifiedGCRPACREBoundedChainConfig,
) -> GCRPACREBoundedAuthorization:
    if specification != FROZEN_INPUT_FACTORY:
        raise PermissionError(
            "bounded CLI accepts only the frozen real input factory "
            f"{FROZEN_INPUT_FACTORY}"
        )
    module_name, separator, attribute_name = specification.partition(":")
    if (
        not separator
        or not module_name
        or not attribute_name
        or "." in attribute_name
    ):
        raise ValueError(
            "input factory must have the form importable.module:function"
        )
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError("input factory attribute is not callable")
    value = factory(chain_config)
    if type(value) is not GCRPACREBoundedAuthorization:
        raise TypeError(
            "input factory must return exact GCRPACREBoundedAuthorization"
        )
    if value.chain_config is not chain_config:
        raise PermissionError(
            "factory authorization is not bound to the CLI chain capability"
        )
    if type(value.evaluator) is not FrozenGCRPACREDREvaluator:
        raise TypeError(
            "bounded CLI accepts only the fixed concrete D_R evaluator"
        )
    value.verify_unchanged()
    return value


def _verify_receipt(
    receipt: dict[str, object],
    authorization: GCRPACREBoundedAuthorization,
):
    evidence = validate_paired_bounded_receipt(
        receipt,
        oof_decision=authorization.oof_decision,
        access_audit=authorization.access_audit,
        full_d_r_cache_artifact=(
            authorization.full_d_r_cache_artifact
        ),
        dataset_free_receipt_fingerprint=(
            authorization.dataset_free_receipt_fingerprint
        ),
        d_r_structural_receipt_fingerprint=(
            authorization.d_r_structural_receipt_fingerprint
        ),
        repository_root=REPOSITORY_ROOT,
    )
    return decide_paired_bounded400(evidence)


def _final_source_closure_audit() -> dict[str, object]:
    """Audit only after every bounded runtime dependency has been imported."""

    assert_gcr_pacre_v24_loaded_source_closure_complete()
    audit = audit_gcr_pacre_v24_loaded_source_closure()
    if audit.get("missing_count") != 0 or audit.get("passed") is not True:
        raise RuntimeError("bounded subprocess source closure is incomplete")
    return audit


def run_once(args: argparse.Namespace) -> dict[str, object]:
    chain_config = load_and_verify_gcr_pacre_bounded_chain_config(
        args.chain_config
    )
    authorization = _load_authorization(
        args.input_factory,
        chain_config=chain_config,
    )
    output = Path(args.output)
    authorization_path = Path(args.authorization_out)
    if not output.is_absolute() or not authorization_path.is_absolute():
        raise ValueError("output and authorization-out must be absolute")
    if (
        output != Path(authorization.output_directory)
        or authorization_path
        != Path(
            str(chain_config.payload["authorization_artifact_path"])
        )
        or str(args.device) != authorization.requested_device
    ):
        raise PermissionError(
            "bounded output/authorization/device differ from chain config"
        )
    normalized_device = _preflight_execution_device(args.device)
    # This is the last fail-closed check before the persistent O_EXCL marker
    # burns the sole bounded attempt.  The fixed factory has now imported and
    # verified every real runtime predecessor.
    _final_source_closure_audit()
    run_start_token = create_gcr_pacre_bounded_run_start_marker(
        authorization
    )
    atomic_write_new_canonical_json(
        authorization_path,
        authorization.canonical_payload(),
    )
    result = run_gcr_pacre_paired_bounded_400(
        authorization,
        run_start_token=run_start_token,
        output_directory=output,
        device=normalized_device,
    )
    receipt = build_paired_bounded_receipt(result)
    atomic_write_new_canonical_json(
        Path(str(chain_config.payload["result_artifact_path"])),
        receipt,
    )
    atomic_write_new_canonical_json(
        Path(str(chain_config.payload["diagnostics_artifact_path"])),
        result.diagnostic_payload,
    )
    decision = _verify_receipt(receipt, authorization)
    atomic_write_new_canonical_json(
        Path(str(chain_config.payload["decision_artifact_path"])),
        decision.payload,
    )
    return {
        "mode": "run-once",
        "authorization_path": str(
            authorization_path.resolve(strict=True)
        ),
        "output": str(output.resolve(strict=True)),
        "bounded_receipt_fingerprint": (
            decision.bounded_receipt_fingerprint
        ),
        "decision_fingerprint": decision.decision_fingerprint,
        "gate_passed": decision.payload["gate_passed"],
        "epochs": 10,
        "steps_per_epoch": 40,
        "updates_per_arm": 400,
        "source_closure_audit": _final_source_closure_audit(),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def verify_only(args: argparse.Namespace) -> dict[str, object]:
    chain_config = load_and_verify_gcr_pacre_bounded_chain_config(
        args.chain_config
    )
    receipt_path = Path(args.receipt)
    if not receipt_path.is_absolute():
        raise ValueError("receipt must be absolute")
    expected_receipt_path = Path(
        str(chain_config.payload["result_artifact_path"])
    )
    if receipt_path != expected_receipt_path:
        raise PermissionError(
            "receipt differs from the frozen chain result artifact path"
        )
    authorization = _load_authorization(
        args.input_factory,
        chain_config=chain_config,
    )
    receipt = read_canonical_json(receipt_path)
    decision = _verify_receipt(receipt, authorization)
    return {
        "mode": "verify-only",
        "receipt": str(receipt_path.resolve(strict=True)),
        "bounded_receipt_fingerprint": (
            decision.bounded_receipt_fingerprint
        ),
        "decision_fingerprint": decision.decision_fingerprint,
        "gate_passed": decision.payload["gate_passed"],
        "source_closure_audit": _final_source_closure_audit(),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-factory",
        required=True,
        help=(
            "importable module:function returning a verifier-token-bound "
            "GCRPACREBoundedAuthorization"
        ),
    )
    parser.add_argument("--chain-config", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run-once", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--authorization-out")
    parser.add_argument("--receipt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.run_once and (not args.output or not args.authorization_out):
        parser.error("--run-once requires --output and --authorization-out")
    if args.verify_only and not args.receipt:
        parser.error("--verify-only requires --receipt")
    if args.verify_only and (args.output or args.authorization_out):
        parser.error(
            "--verify-only rejects --output and --authorization-out"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    value = run_once(args) if args.run_once else verify_only(args)
    print(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
