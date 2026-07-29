#!/usr/bin/env python3
"""Run or finalize the exact GCR-PACRE v24 Formal800 seed pair.

Official execution uses two independent ``run-seed`` processes, normally on
``cuda:0`` and ``cuda:1``.  A third ``finalize-pair`` process replays the full
predecessor-verification chain, verifies both outer evidence receipts and both
physical cache artifacts, and only then issues the pair receipt.  This avoids
global RNG/determinism-state races between training threads.

The in-process factory contract is:

* ``factory(42, chain_token)`` or ``factory(43, chain_token)`` -> one
  freshly verified authorization bound to that private chain token;
* ``factory(None, chain_token)`` -> exact
  ``{42: authorization, 43: authorization}`` mapping (or a seed-ordered
  two-tuple) for the independent-process pair finalization replay.

Factories must re-verify all upstream receipts in the current process.  A JSON
file from an earlier process is never itself treated as a capability.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite_v24.artifact_io import (
    atomic_write_new_canonical_json,
    read_canonical_json,
)
from cure_lite_v24.fixed_dr_evaluator import FrozenGCRPACREDREvaluator
from cure_lite_v24.formal_artifacts import (
    build_formal_evidence_receipt,
    save_gcr_pacre_formal_schedule_atomic,
    save_gcr_pacre_formal_terminal_atomic,
    validate_and_issue_formal_evidence,
)
from cure_lite_v24.formal_cache_artifacts import (
    verify_formal_cache_pair_independence,
)
from cure_lite_v24.formal_training import (
    GCRPACREFormalAuthorization,
    run_gcr_pacre_formal_800,
)
from cure_lite_v24.source_closure import (
    assert_gcr_pacre_v24_loaded_source_closure_complete,
    audit_gcr_pacre_v24_loaded_source_closure,
)
from cure_lite_v24.formal_run_start import (
    VerifiedGCRPACREFormalChainConfig,
    create_gcr_pacre_formal_run_start_marker,
    load_and_verify_gcr_pacre_formal_chain_config,
)
from tools.gcr_pacre_v24_protocol import (
    verify_formal800_training_independence,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAL_PAIR_FILE = "formal800_pair_receipt.json"
FROZEN_INPUT_FACTORY = (
    "cure_lite_v24.real_input_factory:"
    "build_gcr_pacre_v24_stage_authorization"
)
_SEED_ROLES = {
    42: "primary",
    43: "training_integrity_only",
}


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


def _final_source_closure_audit() -> dict[str, object]:
    """Audit only after every Formal runtime dependency has been imported."""

    assert_gcr_pacre_v24_loaded_source_closure_complete()
    audit = audit_gcr_pacre_v24_loaded_source_closure()
    if audit.get("missing_count") != 0 or audit.get("passed") is not True:
        raise RuntimeError("Formal subprocess source closure is incomplete")
    return audit


def _factory(specification: str):
    if specification != FROZEN_INPUT_FACTORY:
        raise PermissionError(
            "Formal CLI accepts only the frozen real input factory "
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
    value = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(value):
        raise TypeError("input factory attribute is not callable")
    return value


def _validate_authorization(
    value: object,
    *,
    seed: int,
    chain_config: VerifiedGCRPACREFormalChainConfig,
) -> GCRPACREFormalAuthorization:
    if (
        type(value) is not GCRPACREFormalAuthorization
        or (value.seed, value.role) != (seed, _SEED_ROLES[seed])
    ):
        raise TypeError(f"factory returned the wrong seed-{seed} authorization")
    if value.chain_config is not chain_config:
        raise PermissionError(
            "factory authorization is not bound to the CLI chain capability"
        )
    if type(value.evaluator) is not FrozenGCRPACREDREvaluator:
        raise TypeError(
            "formal CLI accepts only the fixed concrete D_R evaluator"
        )
    value.verify_unchanged()
    if seed == 43 and (
        value.canonical_payload()["selection_effect"] != "none"
        or value.canonical_payload()["may_replace_seed42_primary"] is not False
        or value.canonical_payload()["D_V_execution_authorized"] is not False
        or value.canonical_payload()["D_T_execution_authorized"] is not False
    ):
        raise PermissionError("seed43 selection/evaluation firewall changed")
    return value


def _load_one(
    specification: str,
    *,
    seed: int,
    chain_config: VerifiedGCRPACREFormalChainConfig,
) -> GCRPACREFormalAuthorization:
    return _validate_authorization(
        _factory(specification)(seed, chain_config),
        seed=seed,
        chain_config=chain_config,
    )


def _load_pair(
    specification: str,
    *,
    chain_config: VerifiedGCRPACREFormalChainConfig,
) -> tuple[GCRPACREFormalAuthorization, GCRPACREFormalAuthorization]:
    raw = _factory(specification)(None, chain_config)
    if isinstance(raw, Mapping):
        if set(raw) != {42, 43}:
            raise ValueError("Formal input mapping must have exact keys 42 and 43")
        values = (raw[42], raw[43])
    elif isinstance(raw, tuple) and len(raw) == 2:
        values = raw
    else:
        raise TypeError(
            "factory(None) must return a (seed42, seed43) tuple or mapping"
        )
    primary = _validate_authorization(
        values[0],
        seed=42,
        chain_config=chain_config,
    )
    integrity = _validate_authorization(
        values[1],
        seed=43,
        chain_config=chain_config,
    )
    if (
        primary.oof_decision is not integrity.oof_decision
        or primary.bounded_decision is not integrity.bounded_decision
        or primary.cache is integrity.cache
        or primary.schedule is integrity.schedule
    ):
        raise PermissionError(
            "Formal seed pair must share verified decisions and use independent "
            "cache/schedule instances"
        )
    verify_formal_cache_pair_independence(
        primary.cache_artifact,
        integrity.cache_artifact,
    )
    return primary, integrity


def _run_one(
    authorization: GCRPACREFormalAuthorization,
    *,
    output: Path,
    device: str,
):
    if (
        output != Path(authorization.output_directory)
        or str(authorization.requested_device) != str(device)
    ):
        raise PermissionError(
            "Formal CLI output/device differ from the frozen chain config"
        )
    normalized_device = _preflight_execution_device(device)
    # The factory and every predecessor verifier are now loaded.  Fail before
    # the persistent marker and before model/optimizer allocation if even one
    # repository runtime source is outside the frozen closure.
    _final_source_closure_audit()
    run_start_token = create_gcr_pacre_formal_run_start_marker(
        authorization
    )
    atomic_write_new_canonical_json(
        Path(
            str(
                authorization.chain_run_binding[
                    "authorization_artifact_path"
                ]
            )
        ),
        authorization.canonical_payload(),
    )
    schedule_artifact = save_gcr_pacre_formal_schedule_atomic(
        Path(
            str(
                authorization.chain_run_binding[
                    "schedule_artifact_path"
                ]
            )
        ),
        authorization=authorization,
    )
    result = run_gcr_pacre_formal_800(
        authorization,
        run_start_token=run_start_token,
        device=normalized_device,
    )
    terminal = save_gcr_pacre_formal_terminal_atomic(
        Path(
            str(
                authorization.chain_run_binding[
                    "terminal_artifact_directory"
                ]
            )
        ),
        formal_result=result,
    )
    evidence_receipt = build_formal_evidence_receipt(
        result,
        schedule_artifact=schedule_artifact,
        terminal_artifact_receipt=terminal,
    )
    evidence_path = atomic_write_new_canonical_json(
        Path(
            str(
                authorization.chain_run_binding[
                    "evidence_artifact_path"
                ]
            )
        ),
        evidence_receipt,
    )
    verified = validate_and_issue_formal_evidence(
        evidence_receipt,
        authorization=authorization,
        repository_root=REPOSITORY_ROOT,
    )
    return result, verified, evidence_path


def run_seed(arguments: argparse.Namespace) -> dict[str, object]:
    chain_config = load_and_verify_gcr_pacre_formal_chain_config(
        arguments.chain_config
    )
    authorization = _load_one(
        arguments.input_factory,
        seed=arguments.seed,
        chain_config=chain_config,
    )
    output = Path(arguments.output)
    if not output.is_absolute():
        raise ValueError("output must be absolute")
    result, verified, evidence_path = _run_one(
        authorization,
        output=output,
        device=arguments.device,
    )
    if arguments.seed == 43 and (
        result.training_receipt.selection_effect != "none"
        or result.training_receipt.may_replace_seed42_primary is not False
        or result.training_receipt.eligible_for_future_D_V_authorization_after_all_external_prerequisites
        is not False
        or result.training_receipt.eligible_for_future_D_T_authorization_after_all_external_prerequisites
        is not False
    ):
        raise PermissionError("seed43 terminal receipt escaped its firewall")
    return {
        "mode": "run-seed",
        "seed": arguments.seed,
        "role": authorization.role,
        "output": str(output.resolve(strict=True)),
        "evidence": str(evidence_path),
        "formal_receipt_fingerprint": verified.receipt_fingerprint,
        "model_fingerprint": (
            result.training_receipt.final_model_fingerprint
        ),
        "epochs": 800,
        "steps_per_epoch": 40,
        "updates": 32_000,
        "selection_effect": (
            "predeclared_primary"
            if arguments.seed == 42
            else "none"
        ),
        "source_closure_audit": _final_source_closure_audit(),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def finalize_pair(arguments: argparse.Namespace) -> dict[str, object]:
    chain_config = load_and_verify_gcr_pacre_formal_chain_config(
        arguments.chain_config
    )
    primary, integrity = _load_pair(
        arguments.input_factory,
        chain_config=chain_config,
    )
    if (
        Path(arguments.seed42_evidence)
        != Path(str(primary.chain_run_binding["evidence_artifact_path"]))
        or Path(arguments.seed43_evidence)
        != Path(str(integrity.chain_run_binding["evidence_artifact_path"]))
    ):
        raise PermissionError(
            "Formal evidence paths differ from the frozen chain config"
        )
    primary_receipt = read_canonical_json(Path(arguments.seed42_evidence))
    integrity_receipt = read_canonical_json(Path(arguments.seed43_evidence))
    primary_verified = validate_and_issue_formal_evidence(
        primary_receipt,
        authorization=primary,
        repository_root=REPOSITORY_ROOT,
    )
    integrity_verified = validate_and_issue_formal_evidence(
        integrity_receipt,
        authorization=integrity,
        repository_root=REPOSITORY_ROOT,
    )
    pair = verify_formal800_training_independence(
        primary_verified,
        integrity_verified,
    )
    output = Path(arguments.output)
    if not output.is_absolute():
        raise ValueError("output must be absolute")
    if output != Path(str(chain_config.payload["formal_pair_receipt_path"])):
        raise PermissionError(
            "Formal pair output differs from the frozen chain config"
        )
    target = atomic_write_new_canonical_json(output, pair.payload)
    return {
        "mode": "finalize-pair",
        "output": str(target),
        "formal_pair_receipt_fingerprint": pair.receipt_fingerprint,
        "seed42_role": "primary",
        "seed43_role": "training_integrity_only",
        "seed43_selection_effect": "none",
        "source_closure_audit": _final_source_closure_audit(),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    seed = subparsers.add_parser("run-seed")
    seed.add_argument("--input-factory", required=True)
    seed.add_argument("--chain-config", required=True)
    seed.add_argument("--seed", type=int, choices=(42, 43), required=True)
    seed.add_argument("--device", required=True)
    seed.add_argument("--output", required=True)

    finalize = subparsers.add_parser("finalize-pair")
    finalize.add_argument("--input-factory", required=True)
    finalize.add_argument("--chain-config", required=True)
    finalize.add_argument("--seed42-evidence", required=True)
    finalize.add_argument("--seed43-evidence", required=True)
    finalize.add_argument("--output", required=True)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    value = (
        run_seed(arguments)
        if arguments.mode == "run-seed"
        else finalize_pair(arguments)
    )
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
