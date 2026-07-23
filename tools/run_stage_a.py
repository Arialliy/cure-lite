#!/usr/bin/env python3
"""Run one model-independent CURE-Lite Stage-A development experiment.

The command consumes standard D_R and D_V probability/feature caches. It
imports no detector implementation, has no official-test argument, and always
writes to a new output directory.
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
from cure_lite.experiment.stage_a_runner import (  # noqa: E402
    StageARunConfig,
    run_stage_a_from_base_caches,
)
from cure_lite.reference_base import (  # noqa: E402
    load_verified_reference_base_run_identity,
)
from cure_lite.stage_a import STAGE_A_METHOD_ORDER  # noqa: E402
from cure_lite.splits import load_and_validate_manifest  # noqa: E402


SUMMARY_SCHEMA = "cure-lite-stage-a-summary-v4"
METHOD_ORDER = STAGE_A_METHOD_ORDER
_RESULT_FIELDS = (
    "pd",
    "miou",
    "niou",
    "pixel_fa",
    "fp_components_per_mp",
    "raw_background_fa",
    "retention",
    "budget_violation",
)
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
    """Emit bounded, machine-readable progress without contaminating stdout."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--d-r-base-index", type=Path, required=True)
    parser.add_argument("--d-v-base-index", type=Path, required=True)
    parser.add_argument("--reference-base-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def load_stage_a_config(path: Path) -> StageARunConfig:
    payload = _json_object(path, name="Stage-A config")
    config = StageARunConfig.from_mapping(payload)
    if config.canonical_payload() != payload:
        raise ValueError("Stage-A config is not canonical")
    return config


def _new_output_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or candidate.exists():
        raise FileExistsError(f"Stage-A output already exists: {candidate}")
    return candidate.resolve(strict=False)


def _metric_summary(metrics: object) -> dict[str, object]:
    return {field: getattr(metrics, field) for field in _RESULT_FIELDS}


def _development_mechanism_screen(results: object) -> dict[str, object]:
    """Apply the predeclared single-seed D_V mechanism-screen rule."""

    base_at_budget = results.base_at_budget
    factual_only = results.factual_only
    factual_exposure_matched = results.factual_exposure_matched
    uniform_legal = results.uniform_legal
    all_within_constraints = not any(
        result.budget_violation
        for result in (
            results.anchor,
            base_at_budget,
            factual_only,
            factual_exposure_matched,
            uniform_legal,
        )
    )
    delta_pd_base = uniform_legal.pd - base_at_budget.pd
    delta_pd_factual = uniform_legal.pd - factual_only.pd
    delta_pd_exposure_matched = uniform_legal.pd - factual_exposure_matched.pd
    pd_rule_met = (
        delta_pd_base > 0.0
        and delta_pd_factual > 0.0
        and delta_pd_exposure_matched > 0.0
    )
    secondary_iou_non_degradation = (
        uniform_legal.miou
        >= max(
            base_at_budget.miou,
            factual_only.miou,
            factual_exposure_matched.miou,
        )
        and uniform_legal.niou
        >= max(
            base_at_budget.niou,
            factual_only.niou,
            factual_exposure_matched.niou,
        )
    )
    signal = all_within_constraints and pd_rule_met
    return {
        "schema_version": "cure-lite-stage-a-development-screen-v2",
        "primary_metric": "total_pd",
        "strict_improvement_required": True,
        "comparators": ["Base@B", "F", "F×"],
        "all_methods_within_constraints": all_within_constraints,
        "u_minus_base_at_budget_pd": delta_pd_base,
        "u_minus_factual_only_pd": delta_pd_factual,
        "u_minus_factual_exposure_matched_pd": delta_pd_exposure_matched,
        "primary_rule_met": pd_rule_met,
        "secondary_iou_non_degradation": secondary_iou_non_degradation,
        "mechanism_signal": signal,
        "interpretation": (
            "supported_single_seed_development_signal"
            if signal
            else "not_supported_single_seed_development_signal"
        ),
        "not_an_independent_generalization_claim": True,
    }


def _summary_payload(completed: object, manifest: object) -> dict[str, object]:
    results = completed.results
    method_results = {
        "A": results.anchor,
        "Base@B": results.base_at_budget,
        "F": results.factual_only,
        "F×": results.factual_exposure_matched,
        "U": results.uniform_legal,
    }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "dataset": manifest.dataset,
        "evaluation_split": "D_V",
        "independent_generalization_result": False,
        "manifest_fingerprint": manifest.fingerprint,
        "complete_fingerprint": completed.complete_fingerprint,
        "training_support": completed.support_summary.canonical_payload(),
        "efficiency": completed.efficiency.canonical_payload(),
        "development_mechanism_screen": _development_mechanism_screen(results),
        "method_order": list(METHOD_ORDER),
        "methods": {
            method: _metric_summary(method_results[method])
            for method in METHOD_ORDER
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = _new_output_path(args.output)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = load_and_validate_manifest(manifest_path)
    cache_contract = load_base_cache_pair_contract(
        args.d_r_base_index,
        args.d_v_base_index,
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

    config = load_stage_a_config(args.config)
    verified_base_identity = load_verified_reference_base_run_identity(
        args.reference_base_run
    )
    completed = run_stage_a_from_base_caches(
        cache_contract.d_r_index_path,
        cache_contract.d_v_index_path,
        d_r_dataset,
        d_v_dataset,
        config,
        output,
        verified_base_identity=verified_base_identity,
        calibration_workers=args.calibration_workers,
        calibration_progress=_calibration_progress,
    )
    print(
        json.dumps(
            _summary_payload(completed, manifest),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
