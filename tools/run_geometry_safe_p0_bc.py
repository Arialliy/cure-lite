#!/usr/bin/env python3
"""Run the create-only D_R geometry-safe P0-B/C follow-on.

The command reconstructs the frozen geometry catalog and zero-copy eligible
view, verifies them against the authoritative P0-A1 artifacts, and then runs
only the frozen common-support and separability diagnostics.  It performs no
training, calibration, inference, candidate-S construction, D_V/D_T
evaluation, or backbone integration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import isfinite
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
from cure_lite.experiment.geometry_catalog_protocol import (  # noqa: E402
    GeometryCatalogProtocol,
    load_geometry_catalog_protocol,
)
from cure_lite.experiment.geometry_safe_catalog import (  # noqa: E402
    GeometrySafeCatalog,
    build_geometry_safe_catalog,
    build_geometry_safe_p0_view,
    build_p0_a1_receipt,
)
from cure_lite.experiment.geometry_safe_p0_bc_protocol import (  # noqa: E402
    GeometrySafeP0BCProtocol,
    load_geometry_safe_p0_bc_protocol,
)
from cure_lite.experiment.p0_protocol import load_p0_config  # noqa: E402
from cure_lite.experiment.p0_support import (  # noqa: E402
    build_p0_b_c_support,
)
from cure_lite.experiment.training_pipeline import (  # noqa: E402
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402


GEOMETRY_SAFE_P0_BC_RUN_SCHEMA = (
    "cure-lite-geometry-safe-p0-bc-run-v1"
)
GEOMETRY_SAFE_P0_BC_POPULATION_SCHEMA = (
    "cure-lite-geometry-safe-p0-bc-population-binding-v1"
)
GEOMETRY_SAFE_P0_BC_GROUP_ACCOUNTING_SCHEMA = (
    "cure-lite-geometry-safe-p0-bc-group-accounting-v1"
)
GEOMETRY_SAFE_P0_BC_DECISION_SCHEMA = (
    "cure-lite-geometry-safe-p0-bc-decision-v1"
)
GEOMETRY_SAFE_P0_B_SUPPORT_SCHEMA = (
    "cure-lite-geometry-safe-p0-b-support-v1"
)
GEOMETRY_SAFE_P0_C_SEPARABILITY_SCHEMA = (
    "cure-lite-geometry-safe-p0-c-separability-v1"
)
GEOMETRY_SAFE_P0_BC_FROZEN_CONFIG_FILE_SHA256 = (
    "3c61a9839c33bd517e86c4c47a48e4b404397f19697e6a1c519948d4081ef047"
)
_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--geometry-catalog-receipt", type=Path, required=True)
    parser.add_argument("--p0-a1-receipt", type=Path, required=True)
    parser.add_argument("--eligible-view-receipt", type=Path, required=True)
    parser.add_argument("--geometry-complete", type=Path, required=True)
    parser.add_argument("--p0-v1-config", type=Path, required=True)
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
            f"geometry-safe P0-B/C output already exists: {absolute}"
        )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError(
                "geometry-safe P0-B/C output may not traverse a symbolic link"
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


def _verify_fingerprinted(
    payload: Mapping[str, Any],
    *,
    name: str,
    field: str = "receipt_fingerprint",
) -> None:
    fingerprint = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
    ):
        raise RuntimeError(f"{name} fingerprint is inconsistent")


def _implementation_binding() -> dict[str, str]:
    paths = (
        _ROOT / "tools" / "run_geometry_safe_p0_bc.py",
        _ROOT / "cure_lite" / "cache" / "base_cache.py",
        _ROOT / "cure_lite" / "cache" / "schema.py",
        _ROOT / "cure_lite" / "cache" / "state_cache.py",
        _ROOT / "cure_lite" / "config.py",
        _ROOT / "cure_lite" / "data.py",
        _ROOT / "cure_lite" / "decoder.py",
        _ROOT / "cure_lite" / "experiment" / "cache_pipeline.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "geometry_catalog_protocol.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "geometry_safe_catalog.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "geometry_safe_p0_bc_protocol.py",
        _ROOT / "cure_lite" / "experiment" / "p0_geometry.py",
        _ROOT / "cure_lite" / "experiment" / "p0_protocol.py",
        _ROOT / "cure_lite" / "experiment" / "p0_support.py",
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
        str(path.relative_to(_ROOT)): file_sha256(path) for path in paths
    }


def _verify_input_binding(
    config: GeometrySafeP0BCProtocol,
    geometry_protocol: GeometryCatalogProtocol,
    manifest_path: Path,
    state_index_path: Path,
    state_index: Mapping[str, Any],
) -> PreprocessConfig:
    binding = config.input_binding
    geometry_binding = geometry_protocol.input_binding
    if binding != geometry_binding:
        raise RuntimeError(
            "P0-B/C input binding differs from geometry-safe P0-A1"
        )
    if file_sha256(manifest_path) != binding.manifest_file_sha256:
        raise RuntimeError("manifest differs from the frozen P0-B/C binding")
    if file_sha256(state_index_path) != binding.state_index_sha256:
        raise RuntimeError(
            "D_R state index differs from the frozen P0-B/C binding"
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
                f"D_R state index {field} differs from P0-B/C config"
            )
    return PreprocessConfig.from_fingerprint_payload(
        state_index.get("preprocessing")
    )


def _verify_statistical_freeze(
    config: GeometrySafeP0BCProtocol,
    p0_v1_config_path: Path,
) -> None:
    upstream = config.upstream_binding
    if (
        file_sha256(p0_v1_config_path)
        != upstream.p0_v1_config_file_sha256
    ):
        raise RuntimeError("P0-v1 config differs from its frozen file binding")
    p0_v1 = load_p0_config(p0_v1_config_path)
    if p0_v1.fingerprint != upstream.p0_v1_config_fingerprint:
        raise RuntimeError(
            "P0-v1 config differs from its frozen protocol fingerprint"
        )
    legacy_payload = p0_v1.canonical_payload()
    current_payload = config.canonical_payload()
    if (
        current_payload["overlap"] != legacy_payload["overlap"]
        or current_payload["separability"] != legacy_payload["separability"]
    ):
        raise RuntimeError(
            "geometry-safe P0-B/C did not preserve the complete "
            "P0-v1 overlap/separability freeze"
        )


def _load_and_verify_upstream(
    config: GeometrySafeP0BCProtocol,
    *,
    geometry_config_path: Path,
    geometry_catalog_path: Path,
    p0_a1_path: Path,
    eligible_view_path: Path,
    geometry_complete_path: Path,
) -> tuple[
    GeometryCatalogProtocol,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    binding = config.upstream_binding
    expected_files = {
        geometry_config_path: binding.geometry_protocol_file_sha256,
        geometry_catalog_path: binding.geometry_catalog_file_sha256,
        p0_a1_path: binding.p0_a1_file_sha256,
        eligible_view_path: binding.eligible_view_file_sha256,
        geometry_complete_path: binding.geometry_complete_file_sha256,
    }
    for path, expected in expected_files.items():
        if file_sha256(path) != expected:
            raise RuntimeError(
                f"upstream geometry-safe file changed: {path.name}"
            )

    geometry_protocol = load_geometry_catalog_protocol(
        geometry_config_path
    )
    if (
        geometry_protocol.fingerprint
        != binding.geometry_protocol_fingerprint
    ):
        raise RuntimeError("upstream geometry protocol fingerprint changed")
    geometry_catalog = _strict_json(
        geometry_catalog_path,
        name="upstream geometry catalog receipt",
    )
    p0_a1 = _strict_json(p0_a1_path, name="upstream P0-A1 receipt")
    eligible_view = _strict_json(
        eligible_view_path,
        name="upstream eligible-view receipt",
    )
    geometry_complete = _strict_json(
        geometry_complete_path,
        name="upstream geometry COMPLETE",
    )
    for payload, name in (
        (geometry_catalog, "upstream geometry catalog receipt"),
        (p0_a1, "upstream P0-A1 receipt"),
        (eligible_view, "upstream eligible-view receipt"),
    ):
        _verify_fingerprinted(payload, name=name)
    _verify_fingerprinted(
        geometry_complete,
        name="upstream geometry COMPLETE",
        field="complete_fingerprint",
    )

    expected_values = (
        (
            geometry_catalog.get("receipt_fingerprint"),
            binding.geometry_catalog_fingerprint,
            "geometry catalog fingerprint",
        ),
        (
            p0_a1.get("receipt_fingerprint"),
            binding.p0_a1_receipt_fingerprint,
            "P0-A1 receipt fingerprint",
        ),
        (
            p0_a1.get("eligible_catalog_fingerprint"),
            binding.eligible_catalog_fingerprint,
            "eligible catalog fingerprint",
        ),
        (
            eligible_view.get("receipt_fingerprint"),
            binding.eligible_view_receipt_fingerprint,
            "eligible-view receipt fingerprint",
        ),
        (
            eligible_view.get("eligible_catalog_fingerprint"),
            binding.eligible_catalog_fingerprint,
            "eligible-view catalog fingerprint",
        ),
        (
            geometry_complete.get("complete_fingerprint"),
            binding.geometry_complete_fingerprint,
            "geometry COMPLETE fingerprint",
        ),
    )
    for actual, expected, name in expected_values:
        if actual != expected:
            raise RuntimeError(f"upstream {name} changed")
    if (
        p0_a1.get("p0_a1_pass") is not True
        or geometry_complete.get("gate_summary", {}).get("A1") is not True
    ):
        raise RuntimeError("upstream P0-A1 is not a passed formal gate")
    receipt_sha = geometry_complete.get("receipt_sha256")
    if not isinstance(receipt_sha, Mapping):
        raise RuntimeError("upstream geometry COMPLETE lacks receipt hashes")
    expected_receipt_hashes = {
        "geometry_catalog.json": binding.geometry_catalog_file_sha256,
        "p0_a1_population_eligibility.json": (
            binding.p0_a1_file_sha256
        ),
        "eligible_view.json": binding.eligible_view_file_sha256,
    }
    if any(
        receipt_sha.get(name) != value
        for name, value in expected_receipt_hashes.items()
    ):
        raise RuntimeError("upstream COMPLETE receipt hashes are inconsistent")
    return (
        geometry_protocol,
        geometry_catalog,
        p0_a1,
        eligible_view,
        geometry_complete,
    )


def _reconstructed_eligible_view_receipt(
    geometry: GeometrySafeCatalog,
    view: object,
    eligible_catalog_fingerprint: str,
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": (
                "cure-lite-geometry-safe-p0-view-validation-v2"
            ),
            "split": "D_R",
            "geometry_catalog_fingerprint": geometry.catalog_fingerprint,
            "eligible_catalog_fingerprint": eligible_catalog_fingerprint,
            "source_ids": list(view.source_ids),
            "source_images": len(view.source_ids),
            "reachable_factual_targets": (
                view.support_summary.reachable_miss_targets
            ),
            "geometry_safe_legal_targets": (
                view.support_summary.decoder_visible_legal_candidates
            ),
            "geometry_safe_synthetic_images": (
                view.support_summary.synthetic_images
            ),
            "candidate_and_example_objects_reused": True,
            "factual_objects_unmodified": True,
            "training_integration": False,
        }
    )


def _population_receipt(
    config: GeometrySafeP0BCProtocol,
    geometry: GeometrySafeCatalog,
    *,
    upstream_p0_a1_fingerprint: str,
) -> dict[str, object]:
    factual = tuple(
        item for item in geometry.factual_records if item.analysis_eligible
    )
    legal = tuple(
        item for item in geometry.legal_records if item.analysis_eligible
    )
    factual_groups = sorted({item.group_id for item in factual})
    legal_groups = sorted({item.group_id for item in legal})
    overlap_groups = sorted(set(factual_groups) & set(legal_groups))
    legal_exclusive_groups = sorted(set(legal_groups) - set(factual_groups))
    overlap_group_set = set(overlap_groups)
    actual = {
        "factual_discovered": (
            len(geometry.factual_records)
            + len(geometry.outside_population_records)
        ),
        "factual_unreachable_outside_population": len(
            geometry.outside_population_records
        ),
        "factual_targets": len(factual),
        "factual_groups": len(factual_groups),
        "legal_candidates_before_geometry_filter": len(
            geometry.legal_records
        ),
        "legal_geometry_excluded": (
            len(geometry.legal_records) - len(legal)
        ),
        "legal_targets": len(legal),
        "legal_source_images": len({item.sample_id for item in legal}),
        "legal_groups": len(legal_groups),
        "role_overlap_groups": len(overlap_groups),
        "role_overlap_factual_targets": sum(
            item.group_id in overlap_group_set for item in factual
        ),
        "role_overlap_legal_targets": sum(
            item.group_id in overlap_group_set for item in legal
        ),
        "legal_exclusive_groups": len(legal_exclusive_groups),
        "group_key": "manifest.group_id",
    }
    expected = asdict(config.population_binding)
    if actual != expected:
        raise RuntimeError(
            "reconstructed geometry-safe population differs from the "
            "frozen P0-B/C population binding"
        )
    return _fingerprinted(
        {
            "schema_version": GEOMETRY_SAFE_P0_BC_POPULATION_SCHEMA,
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "execution_status": "completed",
            "upstream_p0_a1_receipt_fingerprint": (
                upstream_p0_a1_fingerprint
            ),
            "geometry_catalog_fingerprint": (
                geometry.catalog_fingerprint
            ),
            "eligible_catalog_fingerprint": (
                config.upstream_binding.eligible_catalog_fingerprint
            ),
            "counts": actual,
            "group_accounting": {
                "factual_group_ids": factual_groups,
                "legal_group_ids": legal_groups,
                "role_overlap_group_ids": overlap_groups,
                "legal_exclusive_group_ids": legal_exclusive_groups,
                "role_overlap_groups": len(overlap_groups),
                "role_overlap_factual_targets": actual[
                    "role_overlap_factual_targets"
                ],
                "role_overlap_legal_targets": actual[
                    "role_overlap_legal_targets"
                ],
                "legal_exclusive_groups": len(legal_exclusive_groups),
            },
            "same_group_knn_neighbors_excluded": True,
            "grouped_oof_uses_manifest_group_id": True,
            "mmd_removes_role_overlap_from_legal_reference": True,
            "candidate_s_constructed": False,
        }
    )


def _gate_state(payload: Mapping[str, object]) -> str:
    status = payload.get("diagnostic_status")
    if status not in {"pass", "fail", "inconclusive"}:
        raise RuntimeError("P0-B/C diagnostic status is not three-valued")
    return str(status)


def _three_valued_conjunction(states: Sequence[str]) -> str:
    if any(state == "fail" for state in states):
        return "fail"
    if any(state == "inconclusive" for state in states):
        return "inconclusive"
    if all(state == "pass" for state in states):
        return "pass"
    raise RuntimeError("unknown three-valued gate state")


def _p0_b_screening(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, str]]:
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping):
        return "inconclusive", {
            "handcrafted_knn_coverage": "inconclusive",
            "decoder_joint_knn_coverage": "inconclusive",
        }
    states: dict[str, str] = {}
    for output_name, source_name in (
        ("handcrafted_knn_coverage", "handcrafted"),
        ("decoder_joint_knn_coverage", "decoder_joint"),
    ):
        receipt = coverage.get(source_name)
        value = receipt.get("pass") if isinstance(receipt, Mapping) else None
        states[output_name] = (
            "pass" if value is True else "fail" if value is False else "inconclusive"
        )
    return _three_valued_conjunction(tuple(states.values())), states


def _p0_c_screening(
    payload: Mapping[str, object],
    *,
    auc_maximum: float,
) -> tuple[str, dict[str, dict[str, str]]]:
    classifiers = payload.get("grouped_classifier")
    mmd_receipts = payload.get("mmd")
    statuses: dict[str, dict[str, str]] = {}
    space_states: list[str] = []
    for space in ("handcrafted", "decoder_joint"):
        classifier = (
            classifiers.get(space)
            if isinstance(classifiers, Mapping)
            else None
        )
        bootstrap = (
            classifier.get("bootstrap")
            if isinstance(classifier, Mapping)
            else None
        )
        lower = bootstrap.get("lower") if isinstance(bootstrap, Mapping) else None
        upper = bootstrap.get("upper") if isinstance(bootstrap, Mapping) else None
        if (
            isinstance(lower, (int, float))
            and not isinstance(lower, bool)
            and isinstance(upper, (int, float))
            and not isinstance(upper, bool)
            and isfinite(float(lower))
            and isfinite(float(upper))
            and float(lower) <= float(upper)
        ):
            auc_state = (
                "pass"
                if float(upper) <= auc_maximum
                else "fail"
                if float(lower) > auc_maximum
                else "inconclusive"
            )
        else:
            auc_state = "inconclusive"
        mmd = (
            mmd_receipts.get(space)
            if isinstance(mmd_receipts, Mapping)
            else None
        )
        mmd_value = mmd.get("pass") if isinstance(mmd, Mapping) else None
        mmd_state = (
            "pass"
            if mmd_value is True
            else "fail"
            if mmd_value is False
            else "inconclusive"
        )
        space_state = _three_valued_conjunction((auc_state, mmd_state))
        statuses[space] = {
            "auc_bootstrap_interval": auc_state,
            "mmd_against_legal_reference": mmd_state,
            "space_status": space_state,
        }
        space_states.append(space_state)
    return _three_valued_conjunction(tuple(space_states)), statuses


def _fold_fit_audit(
    payload: Mapping[str, object],
) -> dict[str, object]:
    classifiers = payload.get("grouped_classifier")
    if not isinstance(classifiers, Mapping):
        return {
            "status": "inconclusive",
            "reason": "grouped_classifier_receipt_unavailable",
            "spaces": None,
        }
    result: dict[str, object] = {}
    for space in ("handcrafted", "decoder_joint"):
        classifier = classifiers.get(space)
        if not isinstance(classifier, Mapping):
            return {
                "status": "inconclusive",
                "reason": f"{space}_classifier_receipt_unavailable",
                "spaces": None,
            }
        predictions = classifier.get("oof_predictions")
        folds = classifier.get("folds")
        if not isinstance(predictions, list) or not isinstance(folds, list):
            raise RuntimeError(
                f"{space} classifier receipt lacks fold population metadata"
            )
        rows: list[dict[str, object]] = []
        for fold in folds:
            if not isinstance(fold, Mapping):
                raise RuntimeError("classifier fold receipt is not a mapping")
            train_groups = fold.get("train_groups")
            test_groups = fold.get("test_groups")
            if not isinstance(train_groups, list) or not isinstance(
                test_groups, list
            ):
                raise RuntimeError("classifier fold group ledger is invalid")
            train_set = set(train_groups)
            test_set = set(test_groups)
            if train_set & test_set:
                raise RuntimeError("classifier fold train/test groups overlap")
            train_records = [
                item
                for item in predictions
                if isinstance(item, Mapping)
                and item.get("group_id") in train_set
            ]
            test_records = [
                item
                for item in predictions
                if isinstance(item, Mapping)
                and item.get("group_id") in test_set
            ]
            if len(train_records) + len(test_records) != len(predictions):
                raise RuntimeError(
                    "classifier fold does not account for every target"
                )
            projection_records = (
                [
                    item
                    for item in train_records
                    if item.get("role") == "legal"
                ]
                if space == "decoder_joint"
                else []
            )
            projection_identities = [
                {
                    "identity": item["identity"],
                    "group_id": item["group_id"],
                    "role": item["role"],
                }
                for item in projection_records
            ]
            scale_identities = [
                {
                    "identity": item["identity"],
                    "group_id": item["group_id"],
                    "role": item["role"],
                }
                for item in train_records
            ]
            reported_projection = fold.get("projection_fit")
            if space == "handcrafted":
                if reported_projection is not None:
                    raise RuntimeError(
                        "handcrafted fold unexpectedly reports projection fit"
                    )
                projection_audit: dict[str, object] = {
                    "fit_role": "not_applicable",
                    "reported_parameter_fingerprint": None,
                    "population_match": True,
                }
            else:
                if not isinstance(reported_projection, Mapping):
                    raise RuntimeError("joint fold lacks projection fit receipt")
                expected_projection = {
                    "fit_role": "training-fold-legal-targets-only",
                    "fit_targets": len(projection_records),
                    "fit_groups": len(
                        {item["group_id"] for item in projection_records}
                    ),
                    "fit_population_fingerprint": stable_fingerprint(
                        projection_identities
                    ),
                }
                if any(
                    reported_projection.get(field) != value
                    for field, value in expected_projection.items()
                ):
                    raise RuntimeError(
                        "joint projection fit population receipt is inconsistent"
                    )
                projection_unsigned = dict(reported_projection)
                projection_parameter_fingerprint = projection_unsigned.pop(
                    "parameter_fingerprint", None
                )
                if (
                    projection_parameter_fingerprint
                    != stable_fingerprint(projection_unsigned)
                ):
                    raise RuntimeError(
                        "joint projection parameter fingerprint is inconsistent"
                    )
                projection_audit = {
                    **expected_projection,
                    "reported_parameter_fingerprint": (
                        projection_parameter_fingerprint
                    ),
                    "population_match": True,
                    "parameter_fingerprint_match": True,
                }
            reported_scale = fold.get("scale_fit")
            if not isinstance(reported_scale, Mapping):
                raise RuntimeError("classifier fold lacks scale fit receipt")
            expected_scale = {
                "fit_role": "training-fold-all-roles",
                "fit_targets": len(train_records),
                "fit_groups": len(
                    {item["group_id"] for item in train_records}
                ),
                "fit_population_fingerprint": stable_fingerprint(
                    scale_identities
                ),
            }
            if any(
                reported_scale.get(field) != value
                for field, value in expected_scale.items()
            ):
                raise RuntimeError(
                    "classifier scale fit population receipt is inconsistent"
                )
            scale_unsigned = dict(reported_scale)
            scale_parameter_fingerprint = scale_unsigned.pop(
                "parameter_fingerprint", None
            )
            if scale_parameter_fingerprint != stable_fingerprint(scale_unsigned):
                raise RuntimeError(
                    "classifier scale parameter fingerprint is inconsistent"
                )
            classifier_parameters = fold.get("classifier_parameters")
            if not isinstance(classifier_parameters, Mapping):
                raise RuntimeError("classifier parameter receipt is missing")
            classifier_unsigned = dict(classifier_parameters)
            classifier_parameter_fingerprint = classifier_unsigned.pop(
                "parameter_fingerprint", None
            )
            if (
                classifier_parameter_fingerprint
                != stable_fingerprint(classifier_unsigned)
            ):
                raise RuntimeError(
                    "classifier parameter fingerprint is inconsistent"
                )
            scale_audit = {
                **expected_scale,
                "factual_targets": sum(
                    item.get("role") == "factual" for item in train_records
                ),
                "legal_targets": sum(
                    item.get("role") == "legal" for item in train_records
                ),
                "reported_parameter_fingerprint": scale_parameter_fingerprint,
                "population_match": True,
                "parameter_fingerprint_match": True,
            }
            row = {
                "fold": fold.get("fold"),
                "train_groups": len(train_set),
                "train_targets": len(train_records),
                "test_groups": len(test_set),
                "test_targets": len(test_records),
                "projection_fit": projection_audit,
                "scale_fit": scale_audit,
                "classifier_parameter_fingerprint": (
                    classifier_parameter_fingerprint
                ),
                "classifier_parameter_fingerprint_match": True,
            }
            row["fold_fit_fingerprint"] = stable_fingerprint(row)
            rows.append(row)
        result[space] = rows
    return {
        "status": "complete",
        "fingerprint_scope": (
            "fit-populations-and-reported-fold-metadata"
        ),
        "spaces": result,
        "audit_fingerprint": stable_fingerprint(result),
    }


def _decision(
    config: GeometrySafeP0BCProtocol,
    *,
    population: Mapping[str, object],
    p0_b: Mapping[str, object],
    p0_c: Mapping[str, object],
) -> dict[str, object]:
    b_state = _gate_state(p0_b)
    c_state = _gate_state(p0_c)
    p0_a1_b_c_state = _three_valued_conjunction(
        ("pass", b_state, c_state)
    )
    if p0_a1_b_c_state == "pass":
        route = config.decision_policy.pass_route
        reason_codes = ["p0_a1_b_c_passed"]
    elif p0_a1_b_c_state == "fail":
        route = config.decision_policy.fail_route
        reason_codes = ["p0_b_or_p0_c_failed"]
    else:
        route = config.decision_policy.inconclusive_route
        reason_codes = ["p0_b_or_p0_c_inconclusive"]

    # P0-D is deliberately absent, so the full P0 conjunction cannot pass.
    all_p0_state = _three_valued_conjunction(
        ("pass", b_state, c_state, "inconclusive")
    )
    return _fingerprinted(
        {
            "schema_version": GEOMETRY_SAFE_P0_BC_DECISION_SCHEMA,
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "execution_status": "complete",
            "execution_completed": True,
            "decision_inputs": {
                "population_binding_receipt_fingerprint": (
                    population["receipt_fingerprint"]
                ),
                "p0_b_receipt_fingerprint": (
                    p0_b["receipt_fingerprint"]
                ),
                "p0_c_receipt_fingerprint": (
                    p0_c["receipt_fingerprint"]
                ),
                "p0_a1_receipt_fingerprint": (
                    config.upstream_binding.p0_a1_receipt_fingerprint
                ),
                "eligible_catalog_fingerprint": (
                    config.upstream_binding.eligible_catalog_fingerprint
                ),
            },
            "formal_gates": {
                "p0_a1": "pass",
                "p0_b": b_state,
                "p0_c": c_state,
                "p0_d": "not_evaluated",
            },
            "p0_a1_b_c_gate_state": p0_a1_b_c_state,
            "p0_a1_b_c_pass": (
                True
                if p0_a1_b_c_state == "pass"
                else False
                if p0_a1_b_c_state == "fail"
                else None
            ),
            "all_p0_gate_state": all_p0_state,
            "all_p0_pass": None,
            "all_p0_completion_status": "not_complete_p0_d_not_evaluated",
            "all_p0_gate_interpretation": (
                "prefix-failure-can-be-known-while-full-p0-remains-"
                "incomplete"
            ),
            "next_route": route,
            "reason_codes": reason_codes,
            "eligible_to_design_candidate_s": (
                p0_a1_b_c_state == "pass"
            ),
            "eligible_to_construct_candidate_s": False,
            "requires_matched_geometry_safe_uniform_control": (
                config.decision_policy
                .requires_matched_geometry_safe_uniform_control
            ),
            "authorizes_candidate_s_construction": False,
            "authorizes_training": False,
            "authorizes_d_v_evaluation": False,
            "authorizes_full_cure": False,
            "candidate_distribution_constructed": False,
            "p0_d_executed": False,
            "training_performed": False,
            "d_v_accessed": False,
            "d_t_accessed": False,
            "does_not_reinterpret_p0_a0_a1": True,
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
        name="geometry-safe P0-B/C config",
    )
    geometry_config_path = _canonical_existing_file(
        args.geometry_config,
        name="geometry-safe P0-A1 config",
    )
    geometry_catalog_path = _canonical_existing_file(
        args.geometry_catalog_receipt,
        name="geometry catalog receipt",
    )
    p0_a1_path = _canonical_existing_file(
        args.p0_a1_receipt,
        name="P0-A1 receipt",
    )
    eligible_view_path = _canonical_existing_file(
        args.eligible_view_receipt,
        name="eligible-view receipt",
    )
    geometry_complete_path = _canonical_existing_file(
        args.geometry_complete,
        name="geometry COMPLETE",
    )
    p0_v1_config_path = _canonical_existing_file(
        args.p0_v1_config,
        name="P0-v1 config",
    )
    output = _prepare_output(args.output)

    if (
        file_sha256(config_path)
        != GEOMETRY_SAFE_P0_BC_FROZEN_CONFIG_FILE_SHA256
    ):
        raise RuntimeError(
            "geometry-safe P0-B/C config is not the exact frozen file"
        )
    config = load_geometry_safe_p0_bc_protocol(config_path)
    _verify_statistical_freeze(config, p0_v1_config_path)
    (
        geometry_protocol,
        upstream_geometry_catalog,
        upstream_p0_a1,
        upstream_eligible_view,
        _,
    ) = _load_and_verify_upstream(
        config,
        geometry_config_path=geometry_config_path,
        geometry_catalog_path=geometry_catalog_path,
        p0_a1_path=p0_a1_path,
        eligible_view_path=eligible_view_path,
        geometry_complete_path=geometry_complete_path,
    )

    manifest = load_and_validate_manifest(manifest_path)
    if manifest.dataset != config.dataset:
        raise RuntimeError("manifest dataset differs from P0-B/C config")
    state_index = _strict_json(state_index_path, name="D_R state index")
    preprocess = _verify_input_binding(
        config,
        geometry_protocol,
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
    geometry = build_geometry_safe_catalog(
        bundle,
        legacy,
        manifest,
        geometry_protocol,
    )
    reconstructed_geometry_catalog = _fingerprinted(
        geometry.canonical_payload()
    )
    if reconstructed_geometry_catalog != upstream_geometry_catalog:
        raise RuntimeError(
            "reconstructed geometry catalog differs from authoritative A1"
        )
    reconstructed_p0_a1 = _fingerprinted(
        build_p0_a1_receipt(
            geometry,
            geometry_protocol,
            a0_receipt_fingerprint=upstream_p0_a1[
                "a0_receipt_fingerprint"
            ],
        )
    )
    if reconstructed_p0_a1 != upstream_p0_a1:
        raise RuntimeError(
            "reconstructed P0-A1 receipt differs from authoritative A1"
        )
    view = build_geometry_safe_p0_view(legacy, geometry)
    reconstructed_eligible_view = _reconstructed_eligible_view_receipt(
        geometry,
        view,
        config.upstream_binding.eligible_catalog_fingerprint,
    )
    if reconstructed_eligible_view != upstream_eligible_view:
        raise RuntimeError(
            "reconstructed eligible view differs from authoritative A1"
        )
    population = _population_receipt(
        config,
        geometry,
        upstream_p0_a1_fingerprint=upstream_p0_a1[
            "receipt_fingerprint"
        ],
    )

    config_file_sha256 = file_sha256(config_path)
    implementation_files = _implementation_binding()
    immutable_files = {
        str(path): file_sha256(path)
        for path in (
            manifest_path,
            state_index_path,
            config_path,
            geometry_config_path,
            geometry_catalog_path,
            p0_a1_path,
            eligible_view_path,
            geometry_complete_path,
            p0_v1_config_path,
        )
    }

    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / ".incomplete"
    incomplete.touch(exist_ok=False)
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    _write_new_json(receipts / "population_binding.json", population)
    group_accounting = _fingerprinted(
        {
            "schema_version": GEOMETRY_SAFE_P0_BC_GROUP_ACCOUNTING_SCHEMA,
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "execution_status": "completed",
            "population_binding_receipt_fingerprint": (
                population["receipt_fingerprint"]
            ),
            "eligible_catalog_fingerprint": (
                config.upstream_binding.eligible_catalog_fingerprint
            ),
            **population["group_accounting"],
        }
    )
    _write_new_json(receipts / "group_accounting.json", group_accounting)

    p0_b_raw, p0_c_raw, _ = build_p0_b_c_support(
        bundle,
        view,
        manifest,
        config.overlap,
        config.separability,
        formal_eligible=True,
    )
    p0_b_status, p0_b_subitems = _p0_b_screening(p0_b_raw)
    p0_b_pass = (
        True
        if p0_b_status == "pass"
        else False
        if p0_b_status == "fail"
        else None
    )
    p0_b = _fingerprinted(
        {
            "schema_version": GEOMETRY_SAFE_P0_B_SUPPORT_SCHEMA,
            "legacy_raw": p0_b_raw,
            "screening_status": p0_b_status,
            "subitem_status": p0_b_subitems,
            "diagnostic_status": p0_b_status,
            "diagnostic_pass": p0_b_pass,
            "formal_status": p0_b_status,
            "p0_b_pass": p0_b_pass,
            "failure_decision": (
                None
                if p0_b_status == "pass"
                else "redesign_synthetic_state"
                if p0_b_status == "fail"
                else "resolve_p0_bc_inconclusive"
            ),
            "protocol_id": config.protocol_id,
            "execution_status": "completed",
            "formal_gate_role": "p0-b-common-support",
            "population_binding_receipt_fingerprint": (
                population["receipt_fingerprint"]
            ),
            "upstream_p0_a1_receipt_fingerprint": (
                config.upstream_binding.p0_a1_receipt_fingerprint
            ),
            "eligible_catalog_fingerprint": (
                config.upstream_binding.eligible_catalog_fingerprint
            ),
        }
    )
    p0_c_status, p0_c_subitems = _p0_c_screening(
        p0_c_raw,
        auc_maximum=config.separability.auc_maximum,
    )
    p0_c_pass = (
        True
        if p0_c_status == "pass"
        else False
        if p0_c_status == "fail"
        else None
    )
    p0_c = _fingerprinted(
        {
            "schema_version": GEOMETRY_SAFE_P0_C_SEPARABILITY_SCHEMA,
            "legacy_raw": p0_c_raw,
            "screening_status": p0_c_status,
            "subitem_status": p0_c_subitems,
            "fold_fit_audit": _fold_fit_audit(p0_c_raw),
            "diagnostic_status": p0_c_status,
            "diagnostic_pass": p0_c_pass,
            "formal_status": p0_c_status,
            "p0_c_pass": p0_c_pass,
            "failure_decision": (
                None
                if p0_c_status == "pass"
                else "redesign_synthetic_state"
                if p0_c_status == "fail"
                else "resolve_p0_bc_inconclusive"
            ),
            "protocol_id": config.protocol_id,
            "execution_status": "completed",
            "formal_gate_role": "p0-c-strong-separability-and-shift-screen",
            "population_binding_receipt_fingerprint": (
                population["receipt_fingerprint"]
            ),
            "upstream_p0_a1_receipt_fingerprint": (
                config.upstream_binding.p0_a1_receipt_fingerprint
            ),
            "eligible_catalog_fingerprint": (
                config.upstream_binding.eligible_catalog_fingerprint
            ),
            "interpretation_limit": (
                "pass-means-no-strong-shift-detected-by-the-frozen-"
                "probes-not-proof-of-equal-distributions"
            ),
        }
    )
    _write_new_json(receipts / "p0_b_support.json", p0_b)
    _write_new_json(receipts / "p0_c_screening.json", p0_c)
    decision = _decision(
        config,
        population=population,
        p0_b=p0_b,
        p0_c=p0_c,
    )
    _write_new_json(receipts / "decision.json", decision)

    bundle.verify_unchanged()
    if file_sha256(config_path) != config_file_sha256:
        raise RuntimeError("P0-B/C config changed while diagnostics ran")
    if _implementation_binding() != implementation_files:
        raise RuntimeError(
            "P0-B/C implementation changed while diagnostics ran"
        )
    if any(
        file_sha256(Path(path)) != digest
        for path, digest in immutable_files.items()
    ):
        raise RuntimeError("a frozen P0-B/C input changed while diagnostics ran")

    receipt_files = sorted(path.name for path in receipts.iterdir())
    receipt_sha256 = {
        name: file_sha256(receipts / name) for name in receipt_files
    }
    complete = _fingerprinted(
        {
            "schema_version": GEOMETRY_SAFE_P0_BC_RUN_SCHEMA,
            "status": "complete",
            "execution_status": "complete",
            "execution_completed": True,
            "protocol_id": config.protocol_id,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "config_fingerprint": config.fingerprint,
            "config_file_sha256": config_file_sha256,
            "implementation_files": implementation_files,
            "environment": {
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pillow": PIL.__version__,
                "platform": platform.platform(),
            },
            "upstream_p0_a1_receipt_fingerprint": (
                config.upstream_binding.p0_a1_receipt_fingerprint
            ),
            "eligible_catalog_fingerprint": (
                config.upstream_binding.eligible_catalog_fingerprint
            ),
            "population_binding_receipt_fingerprint": (
                population["receipt_fingerprint"]
            ),
            "receipt_files": receipt_files,
            "receipt_sha256": receipt_sha256,
            "decision_fingerprint": decision["receipt_fingerprint"],
            "p0_a1_b_c_gate_state": decision[
                "p0_a1_b_c_gate_state"
            ],
            "all_p0_gate_state": decision["all_p0_gate_state"],
            "all_p0_pass": None,
            "all_p0_completion_status": (
                "not_complete_p0_d_not_evaluated"
            ),
            "next_route": decision["next_route"],
            "eligible_to_design_candidate_s": decision[
                "eligible_to_design_candidate_s"
            ],
            "candidate_distribution_constructed": False,
            "p0_d_executed": False,
            "training_performed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "d_v_accessed": False,
            "d_t_accessed": False,
            "backbone_integration_performed": False,
            "authorizes_training": False,
            "authorizes_candidate_s_construction": False,
            "authorizes_d_v_evaluation": False,
            "authorizes_full_cure": False,
            "does_not_overwrite_p0_a0_a1": True,
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
        "execution_status": "complete",
        "p0_a1_b_c_gate_state": decision["p0_a1_b_c_gate_state"],
        "all_p0_gate_state": decision["all_p0_gate_state"],
        "next_route": decision["next_route"],
        "eligible_to_design_candidate_s": decision[
            "eligible_to_design_candidate_s"
        ],
        "authorizes_training": False,
        "authorizes_d_v_evaluation": False,
        "authorizes_full_cure": False,
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
