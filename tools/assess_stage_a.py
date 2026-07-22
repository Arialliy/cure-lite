#!/usr/bin/env python3
"""Strictly replay and assess one completed CURE-Lite Stage-A run.

The command consumes only the frozen protocol plus D_R/D_V cache and Stage-A
artifacts.  It has no D_T input and writes one new development-assessment JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    _source_tree_digest,
    load_stage_a_run,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402
from tools.run_stage_a import (  # noqa: E402
    METHOD_ORDER,
    _development_mechanism_screen,
    _json_object,
    _metric_summary,
    load_stage_a_config,
)


ASSESSMENT_SCHEMA = "cure-lite-stage-a-assessment-v1"
DECISION_RULE_SCHEMA = "cure-lite-stage-a-decision-rule-v1"
FREEZE_SCHEMA = "cure-lite-stage-a-protocol-freeze-v1"
_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--d-r-base-index", type=Path, required=True)
    parser.add_argument("--d-v-base-index", type=Path, required=True)
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument("--decision-rule", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--stage-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _sha256(path: Path) -> str:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"assessment input may not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"assessment input must be a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_decision_rule(
    rule: Mapping[str, Any],
    *,
    dataset: str,
    seed: int,
    stage_config_sha256: str,
) -> None:
    expected_keys = {
        "all_methods_must_satisfy_configured_constraints",
        "comparators",
        "dataset",
        "evaluation_role",
        "evaluation_split",
        "independent_generalization_claim",
        "method_order",
        "positive_signal_requires_all",
        "primary_metric",
        "schema_version",
        "secondary_quality_metrics",
        "seed",
        "stage_a_config_sha256",
        "strict_improvement_required",
    }
    if set(rule) != expected_keys:
        raise ValueError("Stage-A decision-rule fields are not canonical")
    expected_values = {
        "schema_version": DECISION_RULE_SCHEMA,
        "dataset": dataset,
        "seed": seed,
        "evaluation_role": "development_mechanism_screen",
        "evaluation_split": "D_V",
        "independent_generalization_claim": False,
        "method_order": list(METHOD_ORDER),
        "primary_metric": "total_object_level_pd",
        "comparators": ["Base@B", "F"],
        "strict_improvement_required": True,
        "positive_signal_requires_all": [
            "Pd(U) > Pd(Base@B)",
            "Pd(U) > Pd(F)",
        ],
        "all_methods_must_satisfy_configured_constraints": True,
        "secondary_quality_metrics": ["mIoU", "nIoU"],
        "stage_a_config_sha256": stage_config_sha256,
    }
    if dict(rule) != expected_values:
        raise ValueError("Stage-A decision rule differs from the supported rule")


def _resolve_frozen_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"protocol freeze {name} must be a non-empty path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = _ROOT / candidate
    return candidate.resolve(strict=False)


def validate_protocol_freeze(
    freeze: Mapping[str, Any],
    *,
    manifest_path: Path,
    stage_config_path: Path,
    decision_rule_path: Path,
    d_r_index_path: Path,
    d_v_index_path: Path,
    stage_run_path: Path,
) -> None:
    if freeze.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("unsupported Stage-A protocol-freeze schema")
    if freeze.get("runtime_splits") != ["D_B", "D_R", "D_V"]:
        raise ValueError("protocol freeze runtime split roles changed")
    if freeze.get("unused_split") != "D_T":
        raise ValueError("protocol freeze unused split changed")
    file_bindings = (
        ("manifest_file_sha256", manifest_path),
        ("stage_a_config_sha256", stage_config_path),
        ("decision_rule_sha256", decision_rule_path),
    )
    for field, path in file_bindings:
        if freeze.get(field) != _sha256(path):
            raise RuntimeError(f"protocol freeze {field} no longer matches")
    if freeze.get("method_source_tree_digest") != _source_tree_digest():
        raise RuntimeError("CURE-Lite method source differs from protocol freeze")
    expected_cache = _resolve_frozen_path(freeze.get("cache_output"), name="cache_output")
    if (
        d_r_index_path.resolve(strict=False) != expected_cache / "D_R" / "index.json"
        or d_v_index_path.resolve(strict=False)
        != expected_cache / "D_V" / "index.json"
    ):
        raise ValueError("assessment cache indexes differ from protocol freeze")
    expected_stage = _resolve_frozen_path(
        freeze.get("stage_a_output"), name="stage_a_output"
    )
    if stage_run_path.resolve(strict=False) != expected_stage:
        raise ValueError("assessment Stage-A run differs from protocol freeze")


def _assessment_payload(
    completed: object,
    manifest: object,
    *,
    decision_rule_sha256: str,
    protocol_freeze_sha256: str,
    stage_config_sha256: str,
) -> dict[str, object]:
    results = completed.results
    methods = {
        "A": results.anchor,
        "Base@B": results.base_at_budget,
        "F": results.factual_only,
        "U": results.uniform_legal,
    }
    screen = _development_mechanism_screen(results)
    if not screen["mechanism_signal"]:
        conclusion = "cure_lite_stage_a_mechanism_not_established"
    elif screen["secondary_iou_non_degradation"]:
        conclusion = "cure_lite_stage_a_positive_development_signal"
    else:
        conclusion = "positive_pd_signal_with_secondary_iou_tradeoff"
    calibration = completed.calibration
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "dataset": manifest.dataset,
        "seed": completed.config.training.global_seed,
        "evaluation_split": "D_V",
        "independent_generalization_result": False,
        "runtime_splits": ["D_R", "D_V"],
        "unused_split": "D_T",
        "verified_full_replay": True,
        "manifest_fingerprint": manifest.fingerprint,
        "stage_complete_fingerprint": completed.complete_fingerprint,
        "stage_config_sha256": stage_config_sha256,
        "decision_rule_sha256": decision_rule_sha256,
        "protocol_freeze_sha256": protocol_freeze_sha256,
        "training_support": completed.support_summary.canonical_payload(),
        "selected_thresholds": {
            "A": completed.anchor.selected_threshold,
            "Base@B": calibration.base_at_budget.protocol.selected_threshold,
            "F": calibration.factual_only.protocol.selected_threshold,
            "U": calibration.uniform_legal.protocol.selected_threshold,
        },
        "method_order": list(METHOD_ORDER),
        "methods": {
            method: _metric_summary(methods[method]) for method in METHOD_ORDER
        },
        "development_mechanism_screen": screen,
        "conclusion": conclusion,
        "scope_note": "single-seed D_V development assessment only",
    }


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    candidate = path.expanduser()
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"assessment output already exists: {candidate}")
    resolved = candidate.resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with resolved.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    stage_config_path = args.stage_config.expanduser().resolve(strict=True)
    decision_rule_path = args.decision_rule.expanduser().resolve(strict=True)
    freeze_path = args.protocol_freeze.expanduser().resolve(strict=True)
    stage_run_path = args.stage_run.expanduser().resolve(strict=True)
    config = load_stage_a_config(stage_config_path)
    manifest = load_and_validate_manifest(manifest_path)
    rule = _json_object(decision_rule_path, name="Stage-A decision rule")
    freeze = _json_object(freeze_path, name="Stage-A protocol freeze")
    config_sha256 = _sha256(stage_config_path)
    _validate_decision_rule(
        rule,
        dataset=manifest.dataset,
        seed=config.training.global_seed,
        stage_config_sha256=config_sha256,
    )
    contract = load_base_cache_pair_contract(
        args.d_r_base_index,
        args.d_v_base_index,
    )
    validate_protocol_freeze(
        freeze,
        manifest_path=manifest_path,
        stage_config_path=stage_config_path,
        decision_rule_path=decision_rule_path,
        d_r_index_path=contract.d_r_index_path,
        d_v_index_path=contract.d_v_index_path,
        stage_run_path=stage_run_path,
    )
    preprocess: PreprocessConfig = contract.preprocessing
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
    completed = load_stage_a_run(
        stage_run_path,
        d_r_dataset,
        d_v_dataset,
        expected_base_fingerprint=contract.base_fingerprint,
    )
    if completed.config.canonical_payload() != config.canonical_payload():
        raise RuntimeError("completed Stage-A config differs from frozen config")
    payload = _assessment_payload(
        completed,
        manifest,
        decision_rule_sha256=_sha256(decision_rule_path),
        protocol_freeze_sha256=_sha256(freeze_path),
        stage_config_sha256=config_sha256,
    )
    _write_new_json(args.output, payload)
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
