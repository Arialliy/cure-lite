#!/usr/bin/env python3
"""Run generated preflight or one externally authorized v24 D_R gate.

The real subcommand verifies the sealed dataset-free/efficiency receipt, the
fixed external authorization schema, and the protocol split-access receipt
before importing or invoking the public D_R input builders.  It performs zero
optimizer steps and cannot open D_V or D_T.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import canonical_json
from cure_lite_v24.dr_gate import (
    GCR_PACRE_DR_ACCESS_AUDIT_PATH,
    GCR_PACRE_DR_DATASET_FREE_RECEIPT_R2_PATH,
    GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH,
    GCR_PACRE_DR_RECEIPT_PATH,
    GCR_PACRE_DR_SOURCE_PATHS,
    begin_gcr_pacre_dr_materialization,
    build_gcr_pacre_dr_preaccess_artifacts,
    create_gcr_pacre_dr_run_start_marker,
    required_gcr_pacre_dr_run_start_marker_path,
    run_gcr_pacre_dr_gate,
    run_gcr_pacre_generated_dr_contract_audit,
    verify_and_issue_gcr_pacre_dr_preaccess,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _fixed_path(relative: str) -> Path:
    return (REPOSITORY / relative).absolute()


def _fixed_source_paths() -> dict[str, Path]:
    return {
        name: _fixed_path(relative)
        for name, relative in GCR_PACRE_DR_SOURCE_PATHS.items()
    }


def _new_output_path(raw: str) -> Path:
    candidate = Path(raw).expanduser()
    absolute = Path(os.path.abspath(candidate))
    parent = absolute.parent
    resolved_parent = parent.resolve(strict=True)
    if (
        resolved_parent != parent
        or not parent.is_dir()
        or parent.is_symlink()
    ):
        raise ValueError("output parent must be a canonical directory")
    path = resolved_parent / absolute.name
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    return path


def _fsync_directories(paths: Sequence[Path]) -> None:
    for parent in sorted({path.parent for path in paths}):
        descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directories((path,))
    except BaseException:
        try:
            path.unlink(missing_ok=True)
            _fsync_directories((path,))
        finally:
            raise


def _write_new_json_pair(
    rows: Sequence[tuple[Path, dict[str, object]]],
) -> None:
    if len(rows) != 2 or len({path for path, _ in rows}) != 2:
        raise ValueError("preaccess creation requires two distinct outputs")
    descriptors: list[int | None] = []
    created: list[Path] = []
    try:
        for path, _ in rows:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            descriptors.append(descriptor)
            created.append(path)
        for index, ((_, payload), descriptor) in enumerate(
            zip(rows, descriptors, strict=True)
        ):
            if descriptor is None:
                raise AssertionError("preaccess descriptor disappeared")
            encoded = (canonical_json(payload) + "\n").encode("utf-8")
            with os.fdopen(descriptor, "wb") as handle:
                descriptors[index] = None
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directories(tuple(created))
    except BaseException:
        for descriptor in descriptors:
            if descriptor is not None:
                os.close(descriptor)
        for path in created:
            path.unlink(missing_ok=True)
        if created:
            _fsync_directories(tuple(created))
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GCR-PACRE v24 generated contract audit or authorized "
            "zero-update real D_R structural gate"
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    generated = subparsers.add_parser(
        "generated",
        help="run generated 32/96 structural fixtures only",
    )
    generated.add_argument(
        "--dataset-free-receipt",
        required=True,
    )
    generated.add_argument("--device", default="cpu")
    generated.add_argument("--output", required=True)

    real = subparsers.add_parser(
        "real",
        help="consume one fixed external authorization and open only D_R",
    )
    real.add_argument(
        "--execute-real-dr",
        action="store_true",
        required=True,
        help="explicit one-shot real D_R execution intent",
    )
    real.add_argument("--device", default="cuda:0")

    create = subparsers.add_parser(
        "preaccess-create",
        help=(
            "create exact metadata-only D_R access/authorization files"
        ),
    )

    verify = subparsers.add_parser(
        "preaccess-verify",
        help="verify existing metadata-only D_R preaccess files",
    )
    return parser


def _run_generated(arguments: argparse.Namespace) -> dict[str, object]:
    return run_gcr_pacre_generated_dr_contract_audit(
        dataset_free_receipt_path=arguments.dataset_free_receipt,
        device=arguments.device,
    )


def _run_real(
    arguments: argparse.Namespace,
    *,
    output: Path,
) -> dict[str, object]:
    if arguments.execute_real_dr is not True:
        raise PermissionError("real D_R intent flag is required")

    # This fixed verifier is intentionally called before any D_R tensor
    # builder is invoked.
    source_paths = _fixed_source_paths()
    token = verify_and_issue_gcr_pacre_dr_preaccess(
        dataset_free_receipt_path=_fixed_dataset_free_r2_path(),
        authorization_receipt_path=_fixed_path(
            GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH
        ),
        access_audit_receipt_path=_fixed_path(
            GCR_PACRE_DR_ACCESS_AUDIT_PATH
        ),
    )

    # Complete every failure-prone metadata and device check before the
    # create-only marker burns the sole real-D_R attempt.  The binder reads
    # only the five frozen metadata files; cached tensors remain closed.
    from cure_lite.experiment.coverage_state_real_dr_inputs import (
        bind_coverage_state_real_dr_sources,
    )

    source_binding, _, _, _ = bind_coverage_state_real_dr_sources(
        **source_paths
    )
    if (
        source_binding.binding_fingerprint
        != token.expected_source_binding_fingerprint
    ):
        raise PermissionError("fixed D_R metadata binding changed")

    import torch

    resolved_device = torch.device(arguments.device)
    if resolved_device.type == "cuda":
        if (
            not torch.cuda.is_available()
            or resolved_device.index is None
            or resolved_device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                "requested CUDA device is unavailable before run start"
            )
        # Force context allocation now so a driver/device error cannot first
        # appear after the irreversible marker.
        torch.empty(1, device=resolved_device)
    run_start = create_gcr_pacre_dr_run_start_marker(
        token,
        marker_path=required_gcr_pacre_dr_run_start_marker_path(token),
        requested_device=str(resolved_device),
        requested_receipt_output=output,
    )
    token = begin_gcr_pacre_dr_materialization(token, run_start)

    from cure_lite.experiment.coverage_state_bounded_protocol import (
        COVERAGE_STATE_BOUNDED_SEED,
        build_coverage_state_bounded_population,
    )
    from cure_lite.experiment.coverage_state_real_dr_inputs import (
        build_coverage_state_real_dr_inputs,
    )

    real_inputs = build_coverage_state_real_dr_inputs(
        **source_paths,
    )
    population = build_coverage_state_bounded_population(
        real_inputs.scalar_cache,
        seed=COVERAGE_STATE_BOUNDED_SEED,
    )
    return run_gcr_pacre_dr_gate(
        preaccess_token=token,
        run_start_token=run_start,
        real_inputs=real_inputs,
        bounded_population=population,
        device=str(resolved_device),
    )


def _fixed_dataset_free_r2_path() -> Path:
    return (
        REPOSITORY / GCR_PACRE_DR_DATASET_FREE_RECEIPT_R2_PATH
    )


def _preaccess_summary(
    *,
    access_path: Path,
    authorization_path: Path,
    mode: str,
) -> dict[str, object]:
    token = verify_and_issue_gcr_pacre_dr_preaccess(
        dataset_free_receipt_path=_fixed_dataset_free_r2_path(),
        authorization_receipt_path=authorization_path,
        access_audit_receipt_path=access_path,
    )
    return {
        "mode": mode,
        "access_audit_path": str(access_path),
        "authorization_path": str(authorization_path),
        "access_audit_receipt_fingerprint": (
            token.access_audit_receipt_fingerprint
        ),
        "authorization_fingerprint": (
            token.authorization_fingerprint
        ),
        "dataset_free_receipt_fingerprint": (
            token.dataset_free_receipt_fingerprint
        ),
        "source_closure_fingerprint": (
            token.source_closure_fingerprint
        ),
        "required_run_start_marker_path": str(
            required_gcr_pacre_dr_run_start_marker_path(token)
        ),
        "expected_real_inputs_fingerprint": (
            token.expected_real_inputs_fingerprint
        ),
        "expected_population_fingerprint": (
            token.expected_population_fingerprint
        ),
        "expected_cache_fingerprint": (
            token.expected_cache_fingerprint
        ),
        "D_R_tensor_payload_accessed": False,
        "D_V_tensor_payload_accessed": False,
        "D_T_tensor_payload_accessed": False,
        "training_performed": False,
    }


def _run_preaccess_create(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    del arguments
    access_path = _new_output_path(
        str(_fixed_path(GCR_PACRE_DR_ACCESS_AUDIT_PATH))
    )
    authorization_path = _new_output_path(
        str(_fixed_path(GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH))
    )
    if access_path == authorization_path:
        raise ValueError("preaccess output paths must be distinct")
    access, authorization = (
        build_gcr_pacre_dr_preaccess_artifacts()
    )
    rows = (
        (access_path, access),
        (authorization_path, authorization),
    )
    _write_new_json_pair(rows)
    try:
        return _preaccess_summary(
            access_path=access_path,
            authorization_path=authorization_path,
            mode="preaccess-create",
        )
    except BaseException:
        access_path.unlink(missing_ok=True)
        authorization_path.unlink(missing_ok=True)
        _fsync_directories((access_path, authorization_path))
        raise


def _run_preaccess_verify(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    del arguments
    return _preaccess_summary(
        access_path=_fixed_path(GCR_PACRE_DR_ACCESS_AUDIT_PATH),
        authorization_path=_fixed_path(
            GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH
        ),
        mode="preaccess-verify",
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "preaccess-create":
        print(
            json.dumps(
                _run_preaccess_create(arguments),
                sort_keys=True,
            )
        )
        return 0
    if arguments.mode == "preaccess-verify":
        print(
            json.dumps(
                _run_preaccess_verify(arguments),
                sort_keys=True,
            )
        )
        return 0
    output = _new_output_path(
        arguments.output
        if arguments.mode == "generated"
        else str(_fixed_path(GCR_PACRE_DR_RECEIPT_PATH))
    )
    receipt = (
        _run_generated(arguments)
        if arguments.mode == "generated"
        else _run_real(arguments, output=output)
    )
    _write_new_json(output, receipt)
    decision = receipt.get("decision")
    summary = {
        "output": str(output),
        "receipt_fingerprint": receipt["receipt_fingerprint"],
        "gate_passed": (
            decision.get("gate_passed")
            if isinstance(decision, dict)
            else None
        ),
        "failed_checks": (
            decision.get("failed_checks")
            if isinstance(decision, dict)
            else None
        ),
        "execution_kind": receipt["execution_kind"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
