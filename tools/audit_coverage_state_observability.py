#!/usr/bin/env python3
"""Create a D_R-only raw-catalog and CSLF observability evidence package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.coverage_state_observability import (  # noqa: E402
    CoverageStateObservabilityDecision,
    audit_population_observability,
)
from cure_lite.data import ManifestImageDataset, PreprocessConfig  # noqa: E402
from cure_lite.experiment.cache_pipeline import (  # noqa: E402
    load_d_r_cache_bundle,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (  # noqa: E402
    bind_coverage_state_real_dr_sources,
)
from cure_lite.experiment.coverage_state_raw_catalog import (  # noqa: E402
    build_coverage_state_raw_catalog,
)
from cure_lite.experiment.geometry_safe_catalog import (  # noqa: E402
    build_geometry_safe_catalog,
)
from cure_lite.experiment.training_pipeline import (  # noqa: E402
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402


COVERAGE_STATE_OBSERVABILITY_RUN_SCHEMA = (
    "cure-lite-coverage-state-observability-run-v1"
)
COVERAGE_STATE_OBSERVABILITY_DECISION_SCHEMA = (
    "cure-lite-coverage-state-observability-decision-v1"
)
_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument(
        "--geometry-catalog-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _canonical_file(path: Path, *, name: str) -> Path:
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
            f"coverage-state observability output already exists: {absolute}"
        )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("output path may not traverse a symbolic link")
    return absolute


def _strict_json(path: Path, *, name: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite value {item}")
            ),
        )
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return dict(value)


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


def _implementation_binding() -> dict[str, str]:
    paths = (
        _ROOT / "tools" / "audit_coverage_state_observability.py",
        _ROOT / "cure_lite" / "coverage_state_level_set.py",
        _ROOT / "cure_lite" / "coverage_state_raw_catalog.py",
        _ROOT / "cure_lite" / "coverage_state_observability.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "coverage_state_raw_catalog.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "coverage_state_observability_protocol.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "coverage_state_real_dr_inputs.py",
        _ROOT / "cure_lite" / "coverage_state_sobolev.py",
        _ROOT / "cure_lite" / "experiment" / "cache_pipeline.py",
        _ROOT / "cure_lite" / "experiment" / "geometry_safe_catalog.py",
        _ROOT / "cure_lite" / "experiment" / "training_pipeline.py",
        _ROOT / "cure_lite" / "instances.py",
        _ROOT / "cure_lite" / "intervention.py",
        _ROOT / "cure_lite" / "matching.py",
        _ROOT / "cure_lite" / "paired_types.py",
        _ROOT / "cure_lite" / "splits.py",
        _ROOT / "cure_lite" / "types.py",
    )
    return {
        str(path.relative_to(_ROOT)): file_sha256(path)
        for path in paths
    }


def _validate_bindings(
    *,
    config_path: Path,
    manifest_path: Path,
    state_index_path: Path,
    geometry_config_path: Path,
    geometry_receipt_path: Path,
) -> tuple[
    object,
    object,
    dict[str, object],
    dict[str, object],
    PreprocessConfig,
]:
    (
        _source_binding,
        config,
        geometry_protocol,
        preprocess,
    ) = bind_coverage_state_real_dr_sources(
        manifest_path=manifest_path,
        state_index_path=state_index_path,
        geometry_config_path=geometry_config_path,
        geometry_receipt_path=geometry_receipt_path,
        observability_config_path=config_path,
    )
    binding = config.input_binding
    state_index = _strict_json(
        state_index_path,
        name="D_R state index",
    )
    expected_state = {
        "index_fingerprint": binding.state_index_fingerprint,
        "base_fingerprint": binding.base_fingerprint,
        "base_state_fingerprint": binding.base_state_fingerprint,
        "state_fingerprint": binding.state_fingerprint,
        "gt_fingerprint": binding.gt_fingerprint,
        "dataset": config.dataset,
        "split": "D_R",
    }
    for name, expected in expected_state.items():
        if state_index.get(name) != expected:
            raise RuntimeError(f"D_R state index {name} changed")
    geometry_receipt = _strict_json(
        geometry_receipt_path,
        name="geometry catalog receipt",
    )
    if geometry_receipt.get("receipt_fingerprint") != (
        binding.geometry_catalog_fingerprint
    ):
        raise RuntimeError("geometry catalog receipt fingerprint changed")
    return (
        config,
        geometry_protocol,
        state_index,
        geometry_receipt,
        preprocess,
    )


def _decision_payload(receipt: object) -> dict[str, object]:
    decision = receipt.decision
    if decision is CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF:
        selected = "scalar_max"
        next_route = "build_scalar_cslf_cache_and_fused_step"
    elif decision is CoverageStateObservabilityDecision.AUTHORIZE_PP_CSLF:
        selected = "phase_preserving"
        next_route = "implement_pp_cslf_then_build_cache_and_fused_step"
    elif decision is (
        CoverageStateObservabilityDecision.STATE_TARGET_CONTRACT_UNREALIZABLE
    ):
        selected = None
        next_route = "redesign_scene_state_or_target_contract"
    elif decision is CoverageStateObservabilityDecision.PHASE_RF_UNREACHABLE:
        selected = None
        next_route = "redesign_state_target_or_separate_rf_version"
    else:
        selected = None
        next_route = "repair_raw_population_contract"
    return _fingerprinted(
        {
            "schema_version": (
                COVERAGE_STATE_OBSERVABILITY_DECISION_SCHEMA
            ),
            "split": "D_R",
            "decision": decision.value,
            "selected_representation": selected,
            "next_route": next_route,
            "gate_values": {
                "informative_clean_positive_count": (
                    receipt.informative_clean_positive_count
                ),
                "identity_null_nonidentical_count": (
                    receipt.identity_null_nonidentical_count
                ),
                "scalar_duplicate_input_target_conflicts": (
                    receipt.scalar_duplicate_input_target_conflicts
                ),
                "phase_duplicate_input_target_conflicts": (
                    receipt.phase_duplicate_input_target_conflicts
                ),
                "target_response_outside_scalar_rf_pixels": (
                    receipt.target_response_outside_scalar_rf_pixels
                ),
                "target_response_outside_phase_rf_pixels": (
                    receipt.target_response_outside_phase_rf_pixels
                ),
                "hidden_by_scalar_projection_pairs": (
                    receipt.hidden_by_scalar_projection_pairs
                ),
                "target_response_hidden_only_by_scalar_pixels": (
                    receipt.target_response_hidden_only_by_scalar_pixels
                ),
            },
            "zero_response_scalar_hidden_pair_triggers_pp": False,
            "raw_population_complete": True,
            "training_authorized": False,
            "training_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = _canonical_file(args.manifest, name="manifest")
    state_index_path = _canonical_file(
        args.state_index,
        name="D_R state index",
    )
    geometry_config_path = _canonical_file(
        args.geometry_config,
        name="geometry config",
    )
    geometry_receipt_path = _canonical_file(
        args.geometry_catalog_receipt,
        name="geometry catalog receipt",
    )
    config_path = _canonical_file(
        args.config,
        name="observability config",
    )
    output = _prepare_output(args.output)
    (
        config,
        geometry_protocol,
        _state_index,
        upstream_geometry_receipt,
        preprocess,
    ) = _validate_bindings(
        config_path=config_path,
        manifest_path=manifest_path,
        state_index_path=state_index_path,
        geometry_config_path=geometry_config_path,
        geometry_receipt_path=geometry_receipt_path,
    )
    manifest = load_and_validate_manifest(manifest_path)
    if manifest.dataset != config.dataset:
        raise RuntimeError("manifest dataset differs from observability config")
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
    geometry = build_geometry_safe_catalog(
        bundle,
        legacy,
        manifest,
        geometry_protocol,
    )
    reconstructed_geometry_receipt = _fingerprinted(
        geometry.canonical_payload()
    )
    if reconstructed_geometry_receipt != upstream_geometry_receipt:
        raise RuntimeError("reconstructed geometry catalog differs from receipt")
    raw = build_coverage_state_raw_catalog(bundle, manifest, geometry)
    observability = audit_population_observability(raw)

    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / ".incomplete"
    incomplete.touch(exist_ok=False)
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    config_receipt = _fingerprinted(
        {
            "schema_version": COVERAGE_STATE_OBSERVABILITY_RUN_SCHEMA,
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "config": config.canonical_payload(),
            "config_fingerprint": config.fingerprint,
            "config_file_sha256": file_sha256(config_path),
            "input": {
                "manifest_file_sha256": (
                    bundle.split_manifest_file_sha256
                ),
                "manifest_fingerprint": (
                    bundle.split_manifest_fingerprint
                ),
                "state_index_sha256": bundle.state_index_sha256,
                "state_index_fingerprint": (
                    bundle.state_index_fingerprint
                ),
                "base_fingerprint": bundle.base_fingerprint,
                "base_state_fingerprint": (
                    bundle.base_state_fingerprint
                ),
                "state_fingerprint": bundle.state_fingerprint,
                "gt_fingerprint": bundle.gt_fingerprint,
                "geometry_protocol_config_fingerprint": (
                    geometry_protocol.fingerprint
                ),
                "geometry_catalog_fingerprint": (
                    geometry.catalog_fingerprint
                ),
            },
            "implementation_files": _implementation_binding(),
            "forbidden_operations": [
                "D_V access",
                "D_T access",
                "decoder training",
                "calibration",
                "inference modification",
                "backbone integration",
            ],
        }
    )
    raw_receipt = _fingerprinted(raw.canonical_payload())
    if raw_receipt["receipt_fingerprint"] != raw.catalog_fingerprint:
        raise RuntimeError("raw catalog fingerprint is inconsistent")
    observability_receipt = _fingerprinted(
        observability.canonical_payload()
    )
    if observability_receipt["receipt_fingerprint"] != (
        observability.receipt_fingerprint
    ):
        raise RuntimeError("observability fingerprint is inconsistent")
    decision = _decision_payload(observability)
    receipt_payloads = {
        "config.json": config_receipt,
        "raw_catalog.json": raw_receipt,
        "observability.json": observability_receipt,
        "decision.json": decision,
    }
    for name, payload in receipt_payloads.items():
        _write_new_json(receipts / name, payload)
    receipt_sha256 = {
        name: file_sha256(receipts / name)
        for name in sorted(receipt_payloads)
    }
    complete = {
        "schema_version": COVERAGE_STATE_OBSERVABILITY_RUN_SCHEMA,
        "status": "complete",
        "protocol_id": config.protocol_id,
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "config_fingerprint": config.fingerprint,
        "raw_catalog_fingerprint": raw.catalog_fingerprint,
        "observability_receipt_fingerprint": (
            observability.receipt_fingerprint
        ),
        "decision_fingerprint": decision["receipt_fingerprint"],
        "decision": observability.decision.value,
        "selected_representation": decision["selected_representation"],
        "receipt_files": sorted(receipt_payloads),
        "receipt_sha256": receipt_sha256,
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    complete["complete_fingerprint"] = stable_fingerprint(complete)
    _write_new_json(output / "COMPLETE.json", complete)
    incomplete.unlink()
    bundle.verify_unchanged()
    return complete


def main(argv: Sequence[str] | None = None) -> int:
    complete = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": complete["status"],
                "decision": complete["decision"],
                "selected_representation": (
                    complete["selected_representation"]
                ),
                "raw_catalog_fingerprint": (
                    complete["raw_catalog_fingerprint"]
                ),
                "observability_receipt_fingerprint": (
                    complete["observability_receipt_fingerprint"]
                ),
                "complete_fingerprint": complete["complete_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
