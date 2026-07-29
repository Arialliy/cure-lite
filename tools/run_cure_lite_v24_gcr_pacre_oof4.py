#!/usr/bin/env python3
"""Fixed authorize/run-fold/finalize/verify CLI for v24 real-D_R OOF-4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import canonical_json
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    bind_coverage_state_real_dr_sources,
)
from cure_lite_v24.oof_run_start import (
    authorize_real_oof4_execution_new,
    load_and_verify_real_oof4_execution_authorization,
    required_oof_dr_source_paths,
)
from cure_lite_v24.oof_runner import (
    finalize_real_oof4,
    preflight_oof_execution_device,
    run_real_oof4_fold,
    verify_real_oof4_result_artifact,
)
from cure_lite_v24.source_closure import (
    assert_gcr_pacre_v24_loaded_source_closure_complete,
    audit_gcr_pacre_v24_loaded_source_closure,
)
from tools.gcr_pacre_v24_protocol import (
    verify_oof4_split_preregistration,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPLIT_PREREGISTRATION = (
    REPOSITORY
    / "protocols/IRSTD-1K/gcr_pacre_v24/"
    "D_R_OOF4_split_preregistration.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "v24 GCR-PACRE fixed real-D_R OOF-4 execution; no D_V/D_T "
            "path or fixed uplift threshold exists"
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("authorize")

    run_fold = subparsers.add_parser("run-fold")
    run_fold.add_argument("--fold-id", required=True, type=int)
    run_fold.add_argument("--device", default="cuda:0")
    run_fold.add_argument(
        "--execute-real-dr-oof",
        action="store_true",
        required=True,
    )
    subparsers.add_parser("finalize")
    subparsers.add_parser("verify")
    return parser


def _split():
    return verify_oof4_split_preregistration(
        SPLIT_PREREGISTRATION,
        repository_root=REPOSITORY,
    )


def _bound_context(authorization):
    source = authorization.payload.get("source_binding")
    paths = source.get("paths") if isinstance(source, dict) else None
    if not isinstance(paths, dict):
        raise RuntimeError("persisted OOF authorization lost source paths")
    binding, protocol, geometry, preprocess = (
        bind_coverage_state_real_dr_sources(
            manifest_path=str(paths["manifest"]),
            state_index_path=str(paths["state_index"]),
            geometry_config_path=str(paths["geometry_config"]),
            geometry_receipt_path=str(paths["geometry_receipt"]),
            observability_config_path=str(paths["observability_config"]),
        )
    )
    if binding.binding_fingerprint != authorization.source_binding_fingerprint:
        raise PermissionError("persisted OOF source binding changed")
    return binding, protocol, geometry, preprocess


def _source_closure_audit() -> dict[str, object]:
    assert_gcr_pacre_v24_loaded_source_closure_complete()
    audit = audit_gcr_pacre_v24_loaded_source_closure()
    if audit.get("missing_count") != 0 or audit.get("passed") is not True:
        raise RuntimeError("OOF subprocess source closure is incomplete")
    return audit


def main() -> int:
    arguments = _parser().parse_args()
    split = _split()
    if arguments.mode == "authorize":
        fixed_sources = required_oof_dr_source_paths()
        binding, _, _, _ = bind_coverage_state_real_dr_sources(
            **fixed_sources,
        )
        _source_closure_audit()
        authorization = authorize_real_oof4_execution_new(
            verified_split=split,
            source_binding=binding,
        )
        summary = {
            "mode": "authorize",
            "authorization_path": authorization.artifact_path,
            "authorization_fingerprint": (
                authorization.authorization_fingerprint
            ),
            "runtime_root": authorization.runtime_root,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
    else:
        authorization = (
            load_and_verify_real_oof4_execution_authorization(
                verified_split=split,
            )
        )
        if arguments.mode == "run-fold":
            if arguments.execute_real_dr_oof is not True:
                raise PermissionError("explicit real-D_R OOF intent is required")
            resolved_device = preflight_oof_execution_device(
                arguments.device
            )
            binding, protocol, geometry, preprocess = _bound_context(
                authorization
            )
            # Last fail-closed loaded-module audit before any fold's
            # persistent run-start marker can be created.
            _source_closure_audit()
            result = run_real_oof4_fold(
                fold_id=arguments.fold_id,
                execution_authorization=authorization,
                verified_split=split,
                source_binding=binding,
                protocol=protocol,
                geometry_protocol=geometry,
                preprocess=preprocess,
                available_sample_ids=tuple(split.root_by_sample),
                device=resolved_device,
            )
            summary = {
                "mode": "run-fold",
                "fold_id": result.fold_id,
                "fold_receipt_fingerprint": result.fold_receipt[
                    "receipt_fingerprint"
                ],
                "artifact_paths": list(result.artifact_paths),
                "source_closure_audit": _source_closure_audit(),
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            }
        elif arguments.mode == "finalize":
            _source_closure_audit()
            result = finalize_real_oof4(
                verified_split=split,
                execution_authorization=authorization,
            )
            summary = {
                "mode": "finalize",
                "result_path": result.result_path,
                "decision_fingerprint": (
                    result.decision.decision_fingerprint
                ),
                "gate_passed": result.decision["gate_passed"],
                "fixed_relative_uplift_threshold": None,
                "source_closure_audit": _source_closure_audit(),
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            }
        else:
            _source_closure_audit()
            decision = verify_real_oof4_result_artifact(
                verified_split=split,
                execution_authorization=authorization,
            )
            summary = {
                "mode": "verify",
                "decision_fingerprint": decision.decision_fingerprint,
                "gate_passed": decision["gate_passed"],
                "fixed_relative_uplift_threshold": None,
                "source_closure_audit": _source_closure_audit(),
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            }
    if "source_closure_audit" not in summary:
        summary["source_closure_audit"] = _source_closure_audit()
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
