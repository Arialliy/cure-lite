#!/usr/bin/env python3
"""Extend one completed CURE-Lite Stage-A run with only the M mechanism.

The command reads the historical D_R/D_V cache indexes from the completed
reference run, reconstructs the exact manifest-backed datasets with their
recorded preprocessing, and writes a new create-only M extension.  It has no
resume path and never writes into the historical Stage-A directory.
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

from cure_lite.data import ManifestImageDataset, PreprocessConfig  # noqa: E402
from cure_lite.experiment.cache_pipeline import (  # noqa: E402
    load_base_cache_pair_contract,
)
from cure_lite.experiment.stage_a_m_runner import (  # noqa: E402
    STAGE_A_M_METHOD_ORDER,
    PublishedStageAMExtension,
    run_stage_a_m_extension,
)
from cure_lite.experiment.training_pipeline import (  # noqa: E402
    FixedEpochTrainingLog,
)
from cure_lite.metrics import FORMAL_STAGE_A_METRIC_FIELDS  # noqa: E402
from cure_lite.splits import load_and_validate_manifest  # noqa: E402


SUMMARY_SCHEMA = "cure-lite-stage-a-m-summary-v1"
METHOD_ORDER = STAGE_A_M_METHOD_ORDER
FROZEN_TRAINING_EPOCHS = 800
DEFAULT_CALIBRATION_WORKERS = min(24, max(1, os.cpu_count() or 1))


def _positive_int(value: str) -> int:
    try:
        resolved = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if resolved < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return resolved


def _calibration_progress(done: int, total: int) -> None:
    """Emit the same bounded candidate progress used by formal Stage-A."""

    stride = max(1, total // 20)
    if done == 1 or done == total or done % stride == 0:
        print(
            json.dumps(
                {
                    "event": "calibration_candidate_progress",
                    "completed": done,
                    "total": total,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )


def _training_progress(log: FixedEpochTrainingLog) -> None:
    """Emit the first, every tenth, and final epoch of the frozen 800 epochs."""

    epoch = log.epoch + 1
    if (
        epoch == 1
        or epoch == FROZEN_TRAINING_EPOCHS
        or epoch % 10 == 0
    ):
        print(
            json.dumps(
                {
                    "event": "m_training_epoch_progress",
                    "epoch": epoch,
                    "epochs": FROZEN_TRAINING_EPOCHS,
                    "pool_sizes": dict(log.pool_sizes),
                    "metrics": dict(log.metrics),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-stage-a", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument(
        "--calibration-workers",
        type=_positive_int,
        default=DEFAULT_CALIBRATION_WORKERS,
        help=(
            "candidate-evaluation worker processes; execution-only and does "
            "not alter the frozen scientific protocol"
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic-link path")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} root must be a JSON object")
    return dict(payload)


def _exact_methods(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != set(METHOD_ORDER):
        raise ValueError("M extension result methods are not canonical")
    expected_fields = set(FORMAL_STAGE_A_METRIC_FIELDS)
    methods: dict[str, dict[str, object]] = {}
    for method in METHOD_ORDER:
        metrics = value[method]
        if not isinstance(metrics, Mapping) or set(metrics) != expected_fields:
            raise ValueError(f"M extension {method} metrics are not canonical")
        methods[method] = dict(metrics)
    return methods


def _require_frozen_training_horizon(reference: Path) -> None:
    receipt = _json_object(
        reference / "receipts" / "config.json",
        name="historical Stage-A config receipt",
    )
    run_config = receipt.get("run_config")
    training = (
        run_config.get("training")
        if isinstance(run_config, Mapping)
        else None
    )
    epochs = training.get("epochs") if isinstance(training, Mapping) else None
    if epochs != FROZEN_TRAINING_EPOCHS:
        raise ValueError(
            "formal M extension requires the completed 800-epoch reference"
        )


def _summary_payload(
    completed: PublishedStageAMExtension,
    manifest: object,
) -> dict[str, object]:
    results = _json_object(
        completed.root / "receipts" / "results.json",
        name="M extension results receipt",
    )
    gate = _json_object(
        completed.root / "receipts" / "gate.json",
        name="M extension gate receipt",
    )
    complete = _json_object(
        completed.root / "COMPLETE.json",
        name="M extension COMPLETE receipt",
    )
    if results.get("method_order") != list(METHOD_ORDER):
        raise ValueError("M extension result method order is not canonical")
    methods = _exact_methods(results.get("methods"))
    recovery = results.get("recovery_diagnostics")
    if not isinstance(recovery, Mapping) or set(recovery) != {
        "U@historical",
        "M",
    }:
        raise ValueError("M extension recovery diagnostics are not canonical")
    if (
        results.get("results_fingerprint") != completed.results_fingerprint
        or complete.get("complete_fingerprint")
        != completed.complete_fingerprint
        or complete.get("results_fingerprint")
        != completed.results_fingerprint
        or gate.get("results_fingerprint")
        != completed.results_fingerprint
        or gate.get("mechanism_signal") != completed.mechanism_signal
    ):
        raise RuntimeError("M extension summary bindings differ")

    fingerprints = {
        "complete": completed.complete_fingerprint,
        "current_source_tree": complete.get("source_tree_digest"),
        "reference_snapshot": completed.reference.snapshot_fingerprint,
        "reference_complete": completed.reference.complete_fingerprint,
        "reference_source_tree": completed.reference.source_tree_digest,
        "reference_results": complete.get("reference_results_fingerprint"),
        "reference_calibration": complete.get(
            "reference_calibration_receipt_fingerprint"
        ),
        "config": complete.get("config_fingerprint"),
        "reference_receipt": complete.get("reference_fingerprint"),
        "alignment_catalog": completed.alignment_catalog_fingerprint,
        "alignment_receipt": complete.get(
            "alignment_receipt_fingerprint"
        ),
        "m_decoder_artifact": completed.m_artifact.artifact_fingerprint,
        "m_decoder_state": completed.m_artifact.decoder_state_fingerprint,
        "m_threshold_receipt": completed.m_calibration.receipt_fingerprint,
        "m_calibration_receipt": complete.get(
            "m_calibration_receipt_fingerprint"
        ),
        "results": completed.results_fingerprint,
        "gate": gate.get("gate_fingerprint"),
        "manifest": complete.get("manifest_fingerprint"),
        "preprocessing": complete.get("preprocessing_fingerprint"),
        "base": complete.get("base_fingerprint"),
        "base_state": complete.get("base_state_fingerprint"),
    }
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in fingerprints.values()
    ):
        raise ValueError("M extension summary fingerprint is invalid")
    return {
        "schema_version": SUMMARY_SCHEMA,
        "dataset": getattr(manifest, "dataset"),
        "evaluation_split": "D_V",
        "independent_generalization_result": False,
        "method_order": list(METHOD_ORDER),
        "methods": methods,
        "recovery_diagnostics": {
            method: dict(recovery[method])
            for method in ("U@historical", "M")
        },
        "development_gate": gate,
        "fingerprints": fingerprints,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reference = args.reference_stage_a.expanduser()
    _require_frozen_training_horizon(reference)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = load_and_validate_manifest(manifest_path)
    cache_contract = load_base_cache_pair_contract(
        reference / "d_r" / "base_cache" / "index.json",
        reference / "d_v" / "base_cache" / "index.json",
    )
    preprocess: PreprocessConfig = cache_contract.preprocessing
    d_r_dataset = ManifestImageDataset(
        manifest,
        "D_R",
        preprocess,
        manifest_path=manifest_path,
    )
    d_v_dataset = ManifestImageDataset(
        manifest,
        "D_V",
        preprocess,
        manifest_path=manifest_path,
    )
    completed = run_stage_a_m_extension(
        reference,
        d_r_dataset,
        d_v_dataset,
        args.output,
        device=args.device,
        calibration_workers=args.calibration_workers,
        calibration_progress=_calibration_progress,
        training_progress=_training_progress,
    )
    print(
        json.dumps(
            _summary_payload(completed, manifest),
            indent=2,
            sort_keys=False,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
