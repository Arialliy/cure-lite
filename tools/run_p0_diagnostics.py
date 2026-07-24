#!/usr/bin/env python3
"""Run create-only CURE-Lite P0 diagnostics using exactly D_R.

The command has no D_V input, performs no decoder training or calibration,
and never integrates the diagnostic candidate marginal sampler into training.
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
from cure_lite.experiment.cache_pipeline import (  # noqa: E402
    load_d_r_cache_bundle,
)
from cure_lite.experiment.exposure_audit import (  # noqa: E402
    build_p0_d_exposure,
)
from cure_lite.experiment.p0_geometry import build_p0_a_geometry  # noqa: E402
from cure_lite.experiment.p0_protocol import (  # noqa: E402
    P0DiagnosticConfig,
    load_p0_config,
)
from cure_lite.experiment.p0_support import (  # noqa: E402
    build_p0_b_c_support,
)
from cure_lite.experiment.training_pipeline import (  # noqa: E402
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402


P0_RUN_SCHEMA = "cure-lite-p0-diagnostic-run-v1"
P0_DECISION_SCHEMA = "cure-lite-p0-decision-v1"
P0_FROZEN_CONFIG_FILE_SHA256 = (
    "6f5dbfe83c3ff385a704f99e16a14ab56202536ea2e150d9950a33f847e7d832"
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
        raise FileExistsError(f"P0 output already exists: {absolute}")
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("P0 output may not traverse a symbolic link")
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
    config: P0DiagnosticConfig,
    manifest_path: Path,
    state_index_path: Path,
    state_index: Mapping[str, Any],
) -> PreprocessConfig:
    binding = config.input_binding
    if file_sha256(manifest_path) != binding.manifest_file_sha256:
        raise RuntimeError("P0 manifest differs from its frozen binding")
    if file_sha256(state_index_path) != binding.state_index_sha256:
        raise RuntimeError("P0 state index differs from its frozen binding")
    expected = {
        "index_fingerprint": binding.state_index_fingerprint,
        "base_fingerprint": binding.base_fingerprint,
        "base_state_fingerprint": binding.base_state_fingerprint,
        "state_fingerprint": binding.state_fingerprint,
        "split": "D_R",
        "dataset": config.dataset,
    }
    for field, value in expected.items():
        if state_index.get(field) != value:
            raise RuntimeError(f"P0 state index {field} differs from config")
    return PreprocessConfig.from_fingerprint_payload(
        state_index.get("preprocessing")
    )


def _implementation_binding() -> dict[str, str]:
    paths = (
        _ROOT / "tools" / "run_p0_diagnostics.py",
        _ROOT / "cure_lite" / "cache" / "base_cache.py",
        _ROOT / "cure_lite" / "cache" / "schema.py",
        _ROOT / "cure_lite" / "cache" / "state_cache.py",
        _ROOT / "cure_lite" / "config.py",
        _ROOT / "cure_lite" / "data.py",
        _ROOT / "cure_lite" / "decoder.py",
        _ROOT / "cure_lite" / "experiment" / "p0_protocol.py",
        _ROOT / "cure_lite" / "experiment" / "p0_geometry.py",
        _ROOT / "cure_lite" / "experiment" / "p0_support.py",
        _ROOT / "cure_lite" / "experiment" / "exposure_audit.py",
        _ROOT / "cure_lite" / "experiment" / "cache_pipeline.py",
        _ROOT / "cure_lite" / "experiment" / "training_pipeline.py",
        _ROOT / "cure_lite" / "instances.py",
        _ROOT / "cure_lite" / "intervention.py",
        _ROOT / "cure_lite" / "matching.py",
        _ROOT / "cure_lite" / "occupancy.py",
        _ROOT / "cure_lite" / "sampling.py",
        _ROOT / "cure_lite" / "splits.py",
        _ROOT / "cure_lite" / "supervision.py",
        _ROOT / "cure_lite" / "train" / "pools.py",
        _ROOT / "cure_lite" / "types.py",
    )
    return {
        str(path.relative_to(_ROOT)): file_sha256(path)
        for path in paths
    }


def _decision(
    p0_a: Mapping[str, object],
    p0_b: Mapping[str, object],
    p0_c: Mapping[str, object],
    p0_d: Mapping[str, object],
) -> dict[str, object]:
    a_pass = p0_a.get("p0_a_pass") is True
    b_value = p0_b.get("p0_b_pass")
    c_value = p0_c.get("p0_c_pass")
    d_value = p0_d.get("p0_d_pass")
    b_pass = b_value is True
    c_pass = c_value is True
    d_pass = d_value is True
    if not a_pass:
        route = "rebuild_synthetic_target_extraction"
        reason = "P0-A failed native-to-evaluation geometry integrity"
    elif not b_pass or not c_pass:
        route = "redesign_synthetic_state"
        reason = "P0-B/P0-C did not establish common decoder-input support"
    elif not d_pass:
        route = "revise_marginal_sampling_distribution"
        reason = "P0-D candidate schedule violated exposure constraints"
    else:
        route = "eligible_to_freeze_candidate_s_before_training"
        reason = "P0-A/B/C/D all passed"
    return _fingerprinted(
        {
            "schema_version": P0_DECISION_SCHEMA,
            "split": "D_R",
            "p0_a_pass": a_pass,
            "p0_b_pass": b_value,
            "p0_c_pass": c_value,
            "p0_d_pass": d_value,
            "all_p0_pass": a_pass and b_pass and c_pass and d_pass,
            "next_route": route,
            "reason": reason,
            "authorizes_s_training": False,
            "authorizes_d_v_evaluation": False,
            "authorizes_full_cure": False,
        }
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = _canonical_existing_file(args.manifest, name="manifest")
    state_index_path = _canonical_existing_file(
        args.state_index,
        name="D_R state index",
    )
    config_path = _canonical_existing_file(args.config, name="P0 config")
    output = _prepare_output(args.output)
    if file_sha256(config_path) != P0_FROZEN_CONFIG_FILE_SHA256:
        raise RuntimeError("P0 config is not the exact frozen protocol file")
    config = load_p0_config(config_path)
    manifest = load_and_validate_manifest(manifest_path)
    if manifest.dataset != config.dataset:
        raise RuntimeError("P0 manifest dataset differs from config")
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
    catalog = prepare_training_catalog(
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
            "schema_version": P0_RUN_SCHEMA,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "config": config.canonical_payload(),
            "config_fingerprint": config.fingerprint,
            "config_file_sha256": config_file_sha256,
            "input": {
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": bundle.split_manifest_file_sha256,
                "manifest_fingerprint": bundle.split_manifest_fingerprint,
                "state_index_path": str(state_index_path),
                "state_index_sha256": bundle.state_index_sha256,
                "state_index_fingerprint": bundle.state_index_fingerprint,
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
            ],
        }
    )
    _write_new_json(receipts / "config.json", config_receipt)

    p0_a = _fingerprinted(
        build_p0_a_geometry(bundle, catalog, config.geometry)
    )
    _write_new_json(receipts / "p0_a_geometry.json", p0_a)

    p0_b_raw, p0_c_raw, matrices = build_p0_b_c_support(
        bundle,
        catalog,
        manifest,
        config.overlap,
        config.separability,
        formal_eligible=p0_a["p0_a_pass"] is True,
    )
    p0_b = _fingerprinted(p0_b_raw)
    p0_c = _fingerprinted(p0_c_raw)
    _write_new_json(receipts / "p0_b_support.json", p0_b)
    _write_new_json(receipts / "p0_c_separability.json", p0_c)

    upstream_pass = bool(
        p0_a["p0_a_pass"]
        and p0_b["p0_b_pass"]
        and p0_c["p0_c_pass"]
    )
    p0_d = _fingerprinted(
        build_p0_d_exposure(
            catalog,
            manifest,
            config.overlap,
            config.exposure,
            upstream_pass=upstream_pass,
            factual_hand=matrices["factual_hand"],
            legal_hand=matrices["legal_hand"],
        )
    )
    _write_new_json(receipts / "p0_d_exposure.json", p0_d)

    decision = _decision(p0_a, p0_b, p0_c, p0_d)
    _write_new_json(receipts / "decision.json", decision)
    bundle.verify_unchanged()
    if file_sha256(config_path) != config_file_sha256:
        raise RuntimeError("P0 config changed while diagnostics were running")
    if _implementation_binding() != implementation_files:
        raise RuntimeError(
            "P0 implementation changed while diagnostics were running"
        )
    receipt_files = sorted(path.name for path in receipts.iterdir())
    receipt_sha256 = {
        name: file_sha256(receipts / name) for name in receipt_files
    }
    complete = _fingerprinted(
        {
            "schema_version": P0_RUN_SCHEMA,
            "status": "complete",
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "config_fingerprint": config.fingerprint,
            "receipt_files": receipt_files,
            "receipt_sha256": receipt_sha256,
            "decision_fingerprint": decision["receipt_fingerprint"],
            "next_route": decision["next_route"],
            "all_p0_pass": decision["all_p0_pass"],
            "s_training_performed": False,
            "d_v_evaluation_performed": False,
        },
        field="complete_fingerprint",
    )
    _write_new_json(output / "COMPLETE.json", complete)
    incomplete.unlink()
    return {
        "output": str(output),
        "config_fingerprint": config.fingerprint,
        "complete_fingerprint": complete["complete_fingerprint"],
        "p0": {
            "A": p0_a["p0_a_pass"],
            "B": p0_b["p0_b_pass"],
            "C": p0_c["p0_c_pass"],
            "D": p0_d["p0_d_pass"],
        },
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
