#!/usr/bin/env python3
"""Run the create-only D_R geometry-safe P0-v2 A0/A1 protocol.

The command performs no training, calibration, inference, D_V access, or
backbone integration.  P0-B/C/D and candidate S are deliberately absent from
this stage so that A1 is established before any downstream diagnostic runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import PIL
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.data import ManifestImageDataset, PreprocessConfig  # noqa: E402
from cure_lite.experiment.cache_pipeline import load_d_r_cache_bundle  # noqa: E402
from cure_lite.experiment.geometry_catalog_protocol import (  # noqa: E402
    GeometryCatalogProtocol,
    load_geometry_catalog_protocol,
)
from cure_lite.experiment.geometry_safe_catalog import (  # noqa: E402
    build_geometry_safe_catalog,
    build_geometry_safe_p0_view,
    build_p0_a0_receipt,
    build_p0_a1_receipt,
)
from cure_lite.experiment.training_pipeline import (  # noqa: E402
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402


GEOMETRY_SAFE_RUN_SCHEMA = "cure-lite-geometry-safe-p0-run-v2"
GEOMETRY_SAFE_DECISION_SCHEMA = "cure-lite-geometry-safe-p0-decision-v2"
GEOMETRY_SAFE_FROZEN_CONFIG_FILE_SHA256 = (
    "719e956b7c51b2b2c8294699fe26c2d36d5c8190b0d8bb5c1d5665a0f4344558"
)
_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
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


def _canonical_existing_file(path: Path, *, name: str) -> Path:
    candidate = path.expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if resolved != absolute or not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must be a canonical regular file")
    return resolved


def _prepare_output(path: Path) -> Path:
    candidate = path.expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(
            f"geometry-safe P0 output already exists: {absolute}"
        )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError(
                "geometry-safe P0 output may not traverse a symbolic link"
            )
    return absolute


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=False,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    result = dict(payload)
    if field in result:
        raise ValueError(f"payload already contains {field}")
    result[field] = stable_fingerprint(result)
    return result


def _verify_config_binding(
    config: GeometryCatalogProtocol,
    manifest_path: Path,
    state_index_path: Path,
    state_index: Mapping[str, Any],
) -> PreprocessConfig:
    binding = config.input_binding
    if file_sha256(manifest_path) != binding.manifest_file_sha256:
        raise RuntimeError("geometry manifest differs from its frozen binding")
    if file_sha256(state_index_path) != binding.state_index_sha256:
        raise RuntimeError(
            "geometry D_R state index differs from its frozen binding"
        )
    expected = {
        "index_fingerprint": binding.state_index_fingerprint,
        "base_fingerprint": binding.base_fingerprint,
        "base_state_fingerprint": binding.base_state_fingerprint,
        "state_fingerprint": binding.state_fingerprint,
        "gt_fingerprint": binding.gt_fingerprint,
        "split": "D_R",
        "dataset": config.dataset,
    }
    for field, value in expected.items():
        if state_index.get(field) != value:
            raise RuntimeError(
                f"geometry D_R state index {field} differs from config"
            )
    return PreprocessConfig.from_fingerprint_payload(
        state_index.get("preprocessing")
    )


def _implementation_binding() -> dict[str, str]:
    paths = (
        _ROOT / "tools" / "run_geometry_safe_p0.py",
        _ROOT / "cure_lite" / "cache" / "base_cache.py",
        _ROOT / "cure_lite" / "cache" / "schema.py",
        _ROOT / "cure_lite" / "cache" / "state_cache.py",
        _ROOT / "cure_lite" / "config.py",
        _ROOT / "cure_lite" / "data.py",
        _ROOT / "cure_lite" / "decoder.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "geometry_catalog_protocol.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "geometry_safe_catalog.py",
        _ROOT / "cure_lite" / "experiment" / "p0_geometry.py",
        _ROOT / "cure_lite" / "experiment" / "cache_pipeline.py",
        _ROOT / "cure_lite" / "experiment" / "training_pipeline.py",
        _ROOT / "cure_lite" / "instances.py",
        _ROOT / "cure_lite" / "intervention.py",
        _ROOT / "cure_lite" / "matching.py",
        _ROOT / "cure_lite" / "occupancy.py",
        _ROOT / "cure_lite" / "splits.py",
        _ROOT / "cure_lite" / "supervision.py",
        _ROOT / "cure_lite" / "train" / "pools.py",
        _ROOT / "cure_lite" / "types.py",
    )
    return {
        str(path.relative_to(_ROOT)): file_sha256(path) for path in paths
    }


def _decision(
    *,
    a0: Mapping[str, object],
    a1: Mapping[str, object],
) -> dict[str, object]:
    a1_pass = a1.get("p0_a1_pass") is True
    if a1_pass:
        route = "eligible-to-run-p0-b-c-on-geometry-safe-catalog"
        reason_codes = ["p0_a1_geometry_eligibility_passed"]
    else:
        route = "rebuild-analysis-population-extraction"
        reason_codes = ["p0_a1_geometry_eligibility_failed"]
    return _fingerprinted(
        {
            "schema_version": GEOMETRY_SAFE_DECISION_SCHEMA,
            "split": "D_R",
            "decision_inputs": {
                "p0_a0_receipt_fingerprint": a0["receipt_fingerprint"],
                "p0_a1_receipt_fingerprint": a1["receipt_fingerprint"],
            },
            "non_gating_observations": {
                "p0_a0_dataset_status": a0["audit_status"],
                "dataset_exact_preservation": a0[
                    "dataset_exact_preservation"
                ],
            },
            "formal_gates": {
                "p0_a1_pass": a1["p0_a1_pass"],
                "p0_b_pass": None,
                "p0_c_pass": None,
                "p0_d_pass": None,
            },
            "all_p0_pass": False,
            "eligible_catalog_fingerprint": (
                a1["eligible_catalog_fingerprint"] if a1_pass else None
            ),
            "next_route": route,
            "reason_codes": reason_codes,
            "claim_restrictions": [
                (
                    "analysis-is-limited-to-geometry-eligible-"
                    "evaluation-grid-targets"
                ),
                (
                    "exact-all-native-target-preservation-"
                    "is-not-established"
                ),
            ],
            "requires_matched_uniform_control": True,
            "historical_209_target_u_m_role": "historical-only",
            "eligible_to_run_p0_b_c": a1_pass,
            "eligible_to_freeze_candidate_s": False,
            "authorizes_s_training": False,
            "authorizes_d_v_evaluation": False,
            "authorizes_full_cure": False,
            "does_not_reinterpret_p0_v1": True,
        }
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = _canonical_existing_file(
        args.manifest,
        name="manifest",
    )
    state_index_path = _canonical_existing_file(
        args.state_index,
        name="D_R state index",
    )
    config_path = _canonical_existing_file(
        args.config,
        name="geometry-safe P0 config",
    )
    output = _prepare_output(args.output)
    if file_sha256(config_path) != GEOMETRY_SAFE_FROZEN_CONFIG_FILE_SHA256:
        raise RuntimeError(
            "geometry-safe P0 config is not the exact frozen protocol file"
        )
    config = load_geometry_catalog_protocol(config_path)
    manifest = load_and_validate_manifest(manifest_path)
    if manifest.dataset != config.dataset:
        raise RuntimeError("geometry manifest dataset differs from config")
    state_index = _strict_json(state_index_path, name="D_R state index")
    preprocess = _verify_config_binding(
        config,
        manifest_path,
        state_index_path,
        state_index,
    )
    dataset = ManifestImageDataset(
        manifest,
        "D_R",
        preprocess,
        manifest_path=manifest_path,
    )
    bundle = load_d_r_cache_bundle(
        state_index_path,
        dataset,
        expected_base_fingerprint=config.input_binding.base_fingerprint,
    )
    sources = tuple(
        CachedTrainingSource(
            row.sample_id,
            row.base_output.feature,
            row.base_output.probability,
            row.state,
        )
        for row in bundle.rows
    )
    legacy = prepare_training_catalog(
        sources,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )

    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / ".incomplete"
    incomplete.touch(exist_ok=False)
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    config_file_sha256 = file_sha256(config_path)
    implementation_files = _implementation_binding()

    config_receipt = _fingerprinted(
        {
            "schema_version": GEOMETRY_SAFE_RUN_SCHEMA,
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "config": config.canonical_payload(),
            "config_fingerprint": config.fingerprint,
            "config_file_sha256": config_file_sha256,
            "input": {
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": (
                    bundle.split_manifest_file_sha256
                ),
                "manifest_fingerprint": (
                    bundle.split_manifest_fingerprint
                ),
                "state_index_path": str(state_index_path),
                "state_index_sha256": bundle.state_index_sha256,
                "state_index_fingerprint": (
                    bundle.state_index_fingerprint
                ),
                "base_index_fingerprint": bundle.base_index_fingerprint,
                "base_fingerprint": bundle.base_fingerprint,
                "base_state_fingerprint": bundle.base_state_fingerprint,
                "state_fingerprint": bundle.state_fingerprint,
                "gt_fingerprint": bundle.gt_fingerprint,
            },
            "implementation_files": implementation_files,
            "environment": {
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pillow": PIL.__version__,
                "platform": platform.platform(),
            },
            "forbidden_operations": [
                "D_V access",
                "decoder training",
                "loss modification",
                "calibration",
                "inference modification",
                "backbone integration",
                "P0-B/C/D execution before A1 completion",
                "candidate S construction",
            ],
        }
    )
    _write_new_json(receipts / "config.json", config_receipt)

    geometry = build_geometry_safe_catalog(
        bundle,
        legacy,
        manifest,
        config,
    )
    catalog_receipt = _fingerprinted(geometry.canonical_payload())
    if (
        catalog_receipt["receipt_fingerprint"]
        != geometry.catalog_fingerprint
    ):
        raise RuntimeError("geometry catalog fingerprint is inconsistent")
    _write_new_json(receipts / "geometry_catalog.json", catalog_receipt)

    a0 = _fingerprinted(build_p0_a0_receipt(geometry, config))
    _write_new_json(
        receipts / "p0_a0_dataset_geometry_audit.json",
        a0,
    )
    a1 = _fingerprinted(
        build_p0_a1_receipt(
            geometry,
            config,
            a0_receipt_fingerprint=a0["receipt_fingerprint"],
        )
    )
    _write_new_json(
        receipts / "p0_a1_population_eligibility.json",
        a1,
    )

    if a1["p0_a1_pass"] is True:
        view = build_geometry_safe_p0_view(legacy, geometry)
        view_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-geometry-safe-p0-view-validation-v2"
                ),
                "split": "D_R",
                "geometry_catalog_fingerprint": (
                    geometry.catalog_fingerprint
                ),
                "eligible_catalog_fingerprint": (
                    a1["eligible_catalog_fingerprint"]
                ),
                "source_ids": list(view.source_ids),
                "source_images": len(view.source_ids),
                "reachable_factual_targets": (
                    view.support_summary.reachable_miss_targets
                ),
                "geometry_safe_legal_targets": (
                    view.support_summary
                    .decoder_visible_legal_candidates
                ),
                "geometry_safe_synthetic_images": (
                    view.support_summary.synthetic_images
                ),
                "candidate_and_example_objects_reused": True,
                "factual_objects_unmodified": True,
                "training_integration": False,
            }
        )
    else:
        view_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-geometry-safe-p0-view-validation-v2"
                ),
                "split": "D_R",
                "geometry_catalog_fingerprint": (
                    geometry.catalog_fingerprint
                ),
                "eligible_catalog_fingerprint": None,
                "execution_status": (
                    "not-built-due-to-p0-a1-failure"
                ),
                "training_integration": False,
            }
        )
    _write_new_json(receipts / "eligible_view.json", view_receipt)

    decision = _decision(a0=a0, a1=a1)
    _write_new_json(receipts / "decision.json", decision)
    bundle.verify_unchanged()
    if file_sha256(config_path) != config_file_sha256:
        raise RuntimeError(
            "geometry-safe P0 config changed while diagnostics were running"
        )
    if _implementation_binding() != implementation_files:
        raise RuntimeError(
            "geometry-safe P0 implementation changed while running"
        )
    receipt_files = sorted(path.name for path in receipts.iterdir())
    receipt_sha256 = {
        name: file_sha256(receipts / name) for name in receipt_files
    }
    complete = _fingerprinted(
        {
            "schema_version": GEOMETRY_SAFE_RUN_SCHEMA,
            "status": "complete",
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "config_fingerprint": config.fingerprint,
            "geometry_catalog_fingerprint": (
                geometry.catalog_fingerprint
            ),
            "eligible_catalog_fingerprint": (
                a1["eligible_catalog_fingerprint"]
                if a1["p0_a1_pass"] is True
                else None
            ),
            "receipt_files": receipt_files,
            "receipt_sha256": receipt_sha256,
            "decision_fingerprint": decision["receipt_fingerprint"],
            "next_route": decision["next_route"],
            "gate_summary": {
                "A0": f"{a0['audit_status']}-non-gating",
                "A1": a1["p0_a1_pass"],
                "B": None,
                "C": None,
                "D": None,
            },
            "training_performed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "d_v_evaluation_performed": False,
            "candidate_distribution_constructed": False,
            "s_training_performed": False,
            "does_not_reinterpret_p0_v1": True,
        },
        field="complete_fingerprint",
    )
    _write_new_json(output / "COMPLETE.json", complete)
    incomplete.unlink()
    return {
        "output": str(output),
        "config_fingerprint": config.fingerprint,
        "complete_fingerprint": complete["complete_fingerprint"],
        "A0": a0["audit_status"],
        "A1": a1["p0_a1_pass"],
        "counts": a1["counts"],
        "next_route": decision["next_route"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
