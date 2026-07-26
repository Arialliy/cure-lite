#!/usr/bin/env python3
"""Publish the real D_R paired catalog and frozen seed-42/43 exposure plans.

This create-only command is intentionally narrower than a training runner.
It verifies the frozen paired-objective and geometry-safe input bindings,
reconstructs the real ``D_R`` prepared and geometry catalogs from the cache,
builds the clean/null pair catalog, and replays both complete 800 x 40
two-pair schedules.  It never constructs a dataset for ``D_V`` or ``D_T``
and performs no model forward, optimizer step, calibration, or evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

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
    build_geometry_safe_catalog,
    build_geometry_safe_p0_view,
    build_p0_a1_receipt,
)
from cure_lite.experiment.paired_catalog import build_pair_catalog  # noqa: E402
from cure_lite.experiment.paired_exposure import (  # noqa: E402
    build_paired_exposure_receipt,
)
from cure_lite.experiment.paired_preflight import (  # noqa: E402
    load_pair_preflight_artifact,
    write_pair_preflight_artifact,
)
from cure_lite.experiment.training_pipeline import (  # noqa: E402
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.paired_types import PairCatalog  # noqa: E402
from cure_lite.splits import load_and_validate_manifest  # noqa: E402
from cure_lite.train.paired_pools import build_paired_schedule  # noqa: E402


PAIRED_REAL_RUN_SCHEMA = "cure-lite-real-paired-preflight-run-v1"
PAIRED_REAL_RECEIPT_SCHEMA = "cure-lite-real-paired-preflight-receipt-v1"
PAIRED_PROTOCOL_FINGERPRINT = (
    "5a2f357911fb5f1dc1a946b3dbad429d256c390677d238b2f395fe90ce91fac8"
)
PAIRED_PROTOCOL_FILE_SHA256 = (
    "e4f289a7d960df1c778ae88f20cc66d13e2062194b8300c0b7a257ad20b5c7b2"
)
PAIRED_PROTOCOL_REPO_PATH = (
    "protocols/IRSTD-1K/paired_objective_v1/proposal_receipt.json"
)
GEOMETRY_PROTOCOL_FILE_SHA256 = (
    "719e956b7c51b2b2c8294699fe26c2d36d5c8190b0d8bb5c1d5665a0f4344558"
)
PAIRED_SEEDS = (42, 43)
_ROOT = Path(__file__).resolve().parents[1]
_INCOMPLETE_NAME = ".incomplete"
_COMPLETE_NAME = "COMPLETE.json"
_PAIR_PREFLIGHT_DIR = "pair_preflight"
_RECEIPTS_DIR = "receipts"
_RUN_RECEIPT_NAME = "run_receipt.json"


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
    parser.add_argument("--p0-a1-receipt", type=Path, required=True)
    parser.add_argument("--eligible-view-receipt", type=Path, required=True)
    parser.add_argument("--geometry-complete", type=Path, required=True)
    parser.add_argument("--paired-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")

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
            f"paired real preflight output already exists: {absolute}"
        )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError(
                "paired real preflight output may not traverse a symbolic link"
            )
    return absolute


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
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


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _repo_relative_path(path: Path, *, name: str) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except ValueError as error:
        raise RuntimeError(f"{name} must reside under the project root") from error


def _load_paired_protocol(path: Path) -> dict[str, Any]:
    if _repo_relative_path(
        path,
        name="paired-objective protocol",
    ) != PAIRED_PROTOCOL_REPO_PATH:
        raise RuntimeError(
            "paired protocol path is not the frozen proposal_receipt.json"
        )
    if file_sha256(path) != PAIRED_PROTOCOL_FILE_SHA256:
        raise RuntimeError("paired protocol is not the exact frozen file")
    payload = _strict_json(path, name="paired-objective protocol")
    _verify_fingerprinted(payload, name="paired-objective protocol")
    if payload.get("receipt_fingerprint") != PAIRED_PROTOCOL_FINGERPRINT:
        raise RuntimeError("paired protocol fingerprint differs from the freeze")
    expected = {
        "schema_version": "cure-lite-paired-objective-proposal-v1",
        "dataset": "IRSTD-1K",
        "evidence_split": "D_R",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"paired protocol {field} differs from the freeze")
    schedule = payload.get("future_schedule_contract")
    performance = payload.get("future_performance_gate")
    if not isinstance(schedule, Mapping) or not isinstance(
        performance,
        Mapping,
    ):
        raise RuntimeError("paired protocol schedule/performance gate is malformed")
    frozen_schedule = {
        "epochs": 800,
        "steps_per_epoch": 40,
        "optimizer_updates": 32_000,
        "positive_pairs_per_update": 2,
    }
    for field, value in frozen_schedule.items():
        if schedule.get(field) != value:
            raise RuntimeError(f"paired schedule {field} differs from the freeze")
    if tuple(performance.get("development_seeds", ())) != PAIRED_SEEDS:
        raise RuntimeError("paired development seeds differ from the freeze")
    return payload


def _verify_geometry_input_binding(
    config: GeometryCatalogProtocol,
    manifest_path: Path,
    state_index_path: Path,
    state_index: Mapping[str, Any],
) -> PreprocessConfig:
    binding = config.input_binding
    if file_sha256(manifest_path) != binding.manifest_file_sha256:
        raise RuntimeError("manifest differs from the frozen geometry binding")
    if file_sha256(state_index_path) != binding.state_index_sha256:
        raise RuntimeError("D_R state index differs from the geometry binding")
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
                f"D_R state index {field} differs from the geometry binding"
            )
    return PreprocessConfig.from_fingerprint_payload(
        state_index.get("preprocessing")
    )


def _upstream_binding(
    paired_protocol: Mapping[str, Any],
    *,
    geometry_catalog_path: Path,
    p0_a1_path: Path,
    eligible_view_path: Path,
    geometry_complete_path: Path,
    geometry_protocol: GeometryCatalogProtocol,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    binding = paired_protocol.get("upstream_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("paired protocol upstream binding is malformed")
    expected_files = {
        geometry_catalog_path: binding.get(
            "geometry_catalog_file_sha256"
        ),
        p0_a1_path: binding.get("p0_a1_file_sha256"),
        eligible_view_path: binding.get("eligible_view_file_sha256"),
    }
    for path, expected in expected_files.items():
        if file_sha256(path) != expected:
            raise RuntimeError(
                f"upstream geometry-safe file differs from paired binding: "
                f"{path.name}"
            )
    expected_paths = {
        geometry_catalog_path: binding.get("geometry_catalog_path"),
        p0_a1_path: binding.get("p0_a1_path"),
        eligible_view_path: binding.get("eligible_view_path"),
    }
    for path, expected in expected_paths.items():
        actual = _repo_relative_path(path, name=path.name)
        if actual != expected:
            raise RuntimeError(
                "upstream geometry-safe path differs from paired binding: "
                f"{actual!r} != {expected!r}"
            )

    geometry_catalog = _strict_json(
        geometry_catalog_path,
        name="geometry catalog receipt",
    )
    p0_a1 = _strict_json(p0_a1_path, name="P0-A1 receipt")
    eligible_view = _strict_json(
        eligible_view_path,
        name="eligible-view receipt",
    )
    geometry_complete = _strict_json(
        geometry_complete_path,
        name="geometry COMPLETE",
    )
    for payload, name in (
        (geometry_catalog, "geometry catalog receipt"),
        (p0_a1, "P0-A1 receipt"),
        (eligible_view, "eligible-view receipt"),
    ):
        _verify_fingerprinted(payload, name=name)
    _verify_fingerprinted(
        geometry_complete,
        name="geometry COMPLETE",
        field="complete_fingerprint",
    )

    expected_values = (
        (
            geometry_catalog.get("receipt_fingerprint"),
            binding.get("geometry_catalog_receipt_fingerprint"),
            "geometry catalog fingerprint",
        ),
        (
            p0_a1.get("receipt_fingerprint"),
            binding.get("p0_a1_receipt_fingerprint"),
            "P0-A1 fingerprint",
        ),
        (
            eligible_view.get("receipt_fingerprint"),
            binding.get("eligible_view_receipt_fingerprint"),
            "eligible-view fingerprint",
        ),
        (
            p0_a1.get("eligible_catalog_fingerprint"),
            binding.get("eligible_catalog_fingerprint"),
            "eligible catalog fingerprint",
        ),
        (
            eligible_view.get("eligible_catalog_fingerprint"),
            binding.get("eligible_catalog_fingerprint"),
            "eligible-view catalog fingerprint",
        ),
        (
            geometry_complete.get("config_fingerprint"),
            geometry_protocol.fingerprint,
            "geometry protocol fingerprint",
        ),
        (
            geometry_complete.get("geometry_catalog_fingerprint"),
            binding.get("geometry_catalog_receipt_fingerprint"),
            "geometry COMPLETE catalog fingerprint",
        ),
    )
    for actual, expected, name in expected_values:
        if actual != expected:
            raise RuntimeError(f"upstream {name} differs from the freeze")
    if (
        p0_a1.get("p0_a1_pass") is not True
        or geometry_complete.get("gate_summary", {}).get("A1") is not True
        or geometry_complete.get("split") != "D_R"
        or geometry_complete.get("runtime_splits") != ["D_R"]
    ):
        raise RuntimeError("upstream geometry-safe P0-A1 is not a D_R pass")
    receipt_sha = geometry_complete.get("receipt_sha256")
    if not isinstance(receipt_sha, Mapping):
        raise RuntimeError("geometry COMPLETE lacks receipt hashes")
    for name, path in (
        ("geometry_catalog.json", geometry_catalog_path),
        ("p0_a1_population_eligibility.json", p0_a1_path),
        ("eligible_view.json", eligible_view_path),
    ):
        if receipt_sha.get(name) != file_sha256(path):
            raise RuntimeError(f"geometry COMPLETE hash differs for {name}")
    return geometry_catalog, p0_a1, eligible_view, geometry_complete


def _reconstructed_eligible_view_receipt(
    geometry: object,
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


def _implementation_binding() -> dict[str, str]:
    paths = (
        _ROOT / "tools" / "run_paired_preflight.py",
        _ROOT / "cure_lite" / "cache" / "base_cache.py",
        _ROOT / "cure_lite" / "cache" / "schema.py",
        _ROOT / "cure_lite" / "cache" / "state_cache.py",
        _ROOT / "cure_lite" / "config.py",
        _ROOT / "cure_lite" / "data.py",
        _ROOT / "cure_lite" / "decoder.py",
        _ROOT / "cure_lite" / "experiment" / "cache_pipeline.py",
        _ROOT / "cure_lite" / "experiment" / "geometry_safe_catalog.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "geometry_catalog_protocol.py",
        _ROOT / "cure_lite" / "experiment" / "paired_catalog.py",
        _ROOT / "cure_lite" / "experiment" / "paired_exposure.py",
        _ROOT / "cure_lite" / "experiment" / "paired_preflight.py",
        _ROOT / "cure_lite" / "experiment" / "training_pipeline.py",
        _ROOT / "cure_lite" / "instances.py",
        _ROOT / "cure_lite" / "intervention.py",
        _ROOT / "cure_lite" / "matching.py",
        _ROOT / "cure_lite" / "occupancy.py",
        _ROOT / "cure_lite" / "paired_types.py",
        _ROOT / "cure_lite" / "sampling.py",
        _ROOT / "cure_lite" / "splits.py",
        _ROOT / "cure_lite" / "supervision.py",
        _ROOT / "cure_lite" / "train" / "paired_pools.py",
        _ROOT / "cure_lite" / "train" / "pools.py",
        _ROOT / "cure_lite" / "types.py",
    )
    return {
        str(path.relative_to(_ROOT)): file_sha256(path) for path in paths
    }


def _validate_publication_bindings(
    catalog: PairCatalog,
    input_bindings: Mapping[str, object],
    implementation_files: Mapping[str, str],
) -> None:
    if not isinstance(catalog, PairCatalog) or catalog.split != "D_R":
        raise TypeError("catalog must be a D_R PairCatalog")
    expected = {
        "dataset": catalog.dataset,
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "paired_protocol_fingerprint": PAIRED_PROTOCOL_FINGERPRINT,
        "paired_protocol_file_sha256": PAIRED_PROTOCOL_FILE_SHA256,
        "paired_protocol_repo_path": PAIRED_PROTOCOL_REPO_PATH,
        "geometry_catalog_fingerprint": (
            catalog.geometry_catalog_fingerprint
        ),
        "source_catalog_fingerprint": catalog.source_catalog_fingerprint,
        "manifest_fingerprint": catalog.manifest_fingerprint,
    }
    for field, value in expected.items():
        if input_bindings.get(field) != value:
            raise RuntimeError(
                f"paired publication input binding mismatch: {field}"
            )
    for prefix in ("geometry_catalog", "p0_a1", "eligible_view"):
        if (
            input_bindings.get(f"{prefix}_repo_path")
            != input_bindings.get(f"{prefix}_frozen_repo_path")
            or input_bindings.get(
                f"{prefix}_path_matches_frozen_binding"
            )
            is not True
        ):
            raise RuntimeError(
                f"paired publication input path binding mismatch: {prefix}"
            )
    if catalog.paired_protocol_fingerprint != PAIRED_PROTOCOL_FINGERPRINT:
        raise RuntimeError("pair catalog uses a non-frozen protocol fingerprint")
    for field, value in input_bindings.items():
        if field.endswith("_sha256") or field.endswith("_fingerprint"):
            _require_sha256(value, name=f"input_bindings.{field}")
    if not implementation_files:
        raise ValueError("implementation_files cannot be empty")
    for path, digest in implementation_files.items():
        if not isinstance(path, str) or not path:
            raise ValueError("implementation file path must be non-empty")
        _require_sha256(digest, name=f"implementation_files[{path!r}]")


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {_INCOMPLETE_NAME, _COMPLETE_NAME}
    } | {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted((root / _PAIR_PREFLIGHT_DIR).rglob(_COMPLETE_NAME))
        if path.is_file()
    }


@dataclass(frozen=True)
class PublishedPairedRun:
    root: Path
    catalog_fingerprint: str
    seed42_schedule_fingerprint: str
    seed43_schedule_fingerprint: str
    run_receipt_fingerprint: str
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        verified = load_paired_run_artifact(self.root)
        if verified != self:
            raise RuntimeError("published paired run identity changed")


def write_paired_run_artifact(
    catalog: PairCatalog,
    output_dir: str | Path,
    *,
    input_bindings: Mapping[str, object],
    implementation_files: Mapping[str, str],
) -> PublishedPairedRun:
    """Publish one tensor-free preflight plus both frozen exposure receipts."""

    _validate_publication_bindings(
        catalog,
        input_bindings,
        implementation_files,
    )
    root = _prepare_output(Path(output_dir))
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    incomplete = root / _INCOMPLETE_NAME
    incomplete.open("xb").close()

    preflight = write_pair_preflight_artifact(
        catalog,
        root / _PAIR_PREFLIGHT_DIR,
    )
    receipts = root / _RECEIPTS_DIR
    receipts.mkdir(exist_ok=False)
    exposures: dict[int, dict[str, object]] = {}
    for seed in PAIRED_SEEDS:
        schedule = build_paired_schedule(catalog, seed=seed)
        exposure = build_paired_exposure_receipt(schedule)
        exposures[seed] = exposure
        _write_new_json(
            receipts / f"exposure_seed{seed}.json",
            exposure,
        )

    preflight_receipt = _strict_json(
        preflight.root / "preflight_receipt.json",
        name="published paired preflight receipt",
    )
    preflight_passed = (
        preflight_receipt.get("integrity_gates", {}).get(
            "preflight_passed"
        )
        is True
    )
    exposure_passed = all(
        receipt["gates"]["all_targets_exposed"] is True
        and receipt["gates"]["all_sources_exposed"] is True
        and receipt["gates"][
            "maximum_pair_exposure_count_difference_at_most_one"
        ]
        is True
        and receipt["gates"]["every_update_uses_two_distinct_sources"]
        is True
        for receipt in exposures.values()
    )
    readiness = preflight_passed and exposure_passed
    run_receipt = _fingerprinted(
        {
            "schema_version": PAIRED_REAL_RECEIPT_SCHEMA,
            "execution_status": "completed",
            "dataset": catalog.dataset,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "input_bindings": dict(input_bindings),
            "implementation_files": dict(implementation_files),
            "pair_catalog_fingerprint": catalog.catalog_fingerprint,
            "pair_preflight": {
                "manifest_fingerprint": preflight.manifest_fingerprint,
                "receipt_fingerprint": preflight.receipt_fingerprint,
                "complete_fingerprint": preflight.complete_fingerprint,
            },
            "counts": {
                "clean_positive": len(catalog.clean_positive),
                "component_null": len(catalog.component_null),
                "identity_null": len(catalog.identity_null),
                "exclusions": len(catalog.exclusions),
            },
            "exposure_receipts": {
                str(seed): {
                    "receipt_fingerprint": exposures[seed][
                        "receipt_fingerprint"
                    ],
                    "schedule_fingerprint": exposures[seed][
                        "schedule_fingerprint"
                    ],
                    "sequence_fingerprint": exposures[seed][
                        "sequence_fingerprint"
                    ],
                }
                for seed in PAIRED_SEEDS
            },
            "gates": {
                "pair_preflight_passed": preflight_passed,
                "seed42_exposure_passed": all(
                    value is True
                    for key, value in exposures[42]["gates"].items()
                    if key
                    in {
                        "all_targets_exposed",
                        "all_sources_exposed",
                        (
                            "maximum_pair_exposure_count_"
                            "difference_at_most_one"
                        ),
                        "every_update_uses_two_distinct_sources",
                    }
                ),
                "seed43_exposure_passed": all(
                    value is True
                    for key, value in exposures[43]["gates"].items()
                    if key
                    in {
                        "all_targets_exposed",
                        "all_sources_exposed",
                        (
                            "maximum_pair_exposure_count_"
                            "difference_at_most_one"
                        ),
                        "every_update_uses_two_distinct_sources",
                    }
                ),
                "bounded_d_r_learnability_ready": readiness,
            },
            "next_route": (
                "run_bounded_d_r_only_paired_learnability"
                if readiness
                else "repair_pair_catalog_or_schedule_before_training"
            ),
            "execution_policy": {
                "read_only_splits": ["D_R"],
                "training_performed": False,
                "formal_training_authorized": False,
                "d_v_accessed": False,
                "d_t_accessed": False,
                "model_forward_performed": False,
                "optimizer_step_performed": False,
                "calibration_performed": False,
                "inference_performed": False,
                "full_cure_started": False,
                "backbone_integration_performed": False,
            },
        }
    )
    _write_new_json(receipts / _RUN_RECEIPT_NAME, run_receipt)
    preflight.verify_unchanged()

    artifact_files = _artifact_hashes(root)
    complete = _fingerprinted(
        {
            "schema_version": PAIRED_REAL_RUN_SCHEMA,
            "execution_status": "complete",
            "dataset": catalog.dataset,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "paired_protocol_fingerprint": PAIRED_PROTOCOL_FINGERPRINT,
            "pair_catalog_fingerprint": catalog.catalog_fingerprint,
            "run_receipt_fingerprint": run_receipt[
                "receipt_fingerprint"
            ],
            "seed42_schedule_fingerprint": exposures[42][
                "schedule_fingerprint"
            ],
            "seed43_schedule_fingerprint": exposures[43][
                "schedule_fingerprint"
            ],
            "artifact_files": artifact_files,
            "artifact_file_count": len(artifact_files),
            "training_performed": False,
            "d_v_accessed": False,
            "d_t_accessed": False,
            "next_route": run_receipt["next_route"],
        },
        field="complete_fingerprint",
    )
    _write_new_json(root / _COMPLETE_NAME, complete)
    incomplete.unlink()
    return load_paired_run_artifact(root)


def load_paired_run_artifact(
    output_dir: str | Path,
) -> PublishedPairedRun:
    """Load and verify a completed real paired preflight publication."""

    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("paired run root must be a regular directory")
    if (root / _INCOMPLETE_NAME).exists():
        raise RuntimeError("paired run publication is incomplete")
    if {path.name for path in root.iterdir()} != {
        _PAIR_PREFLIGHT_DIR,
        _RECEIPTS_DIR,
        _COMPLETE_NAME,
    }:
        raise RuntimeError("paired run top-level inventory changed")
    if {
        path.name for path in (root / _RECEIPTS_DIR).iterdir()
    } != {
        _RUN_RECEIPT_NAME,
        "exposure_seed42.json",
        "exposure_seed43.json",
    }:
        raise RuntimeError("paired run receipt inventory changed")

    preflight = load_pair_preflight_artifact(root / _PAIR_PREFLIGHT_DIR)
    complete = _strict_json(root / _COMPLETE_NAME, name="paired COMPLETE")
    _verify_fingerprinted(
        complete,
        name="paired COMPLETE",
        field="complete_fingerprint",
    )
    run_receipt = _strict_json(
        root / _RECEIPTS_DIR / _RUN_RECEIPT_NAME,
        name="paired run receipt",
    )
    _verify_fingerprinted(run_receipt, name="paired run receipt")
    exposures = {
        seed: _strict_json(
            root / _RECEIPTS_DIR / f"exposure_seed{seed}.json",
            name=f"seed-{seed} exposure receipt",
        )
        for seed in PAIRED_SEEDS
    }
    for seed, receipt in exposures.items():
        _verify_fingerprinted(
            receipt,
            name=f"seed-{seed} exposure receipt",
        )
        if (
            receipt.get("seed") != seed
            or receipt.get("evidence_split") != "D_R"
        ):
            raise RuntimeError("paired exposure seed/split binding changed")

    expected_files = _artifact_hashes(root)
    if complete.get("artifact_files") != expected_files:
        raise RuntimeError("paired run artifact hashes changed")
    if complete.get("artifact_file_count") != len(expected_files):
        raise RuntimeError("paired run artifact count changed")
    if not (
        complete.get("paired_protocol_fingerprint")
        == run_receipt.get("input_bindings", {}).get(
            "paired_protocol_fingerprint"
        )
        == PAIRED_PROTOCOL_FINGERPRINT
        and complete.get("pair_catalog_fingerprint")
        == run_receipt.get("pair_catalog_fingerprint")
        == preflight.pair_catalog_fingerprint
        and complete.get("run_receipt_fingerprint")
        == run_receipt.get("receipt_fingerprint")
        and complete.get("seed42_schedule_fingerprint")
        == exposures[42].get("schedule_fingerprint")
        and complete.get("seed43_schedule_fingerprint")
        == exposures[43].get("schedule_fingerprint")
    ):
        raise RuntimeError("paired run cross-file bindings disagree")
    policy = run_receipt.get("execution_policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(field) is not False
        for field in (
            "training_performed",
            "formal_training_authorized",
            "d_v_accessed",
            "d_t_accessed",
            "model_forward_performed",
            "optimizer_step_performed",
            "calibration_performed",
            "inference_performed",
            "full_cure_started",
            "backbone_integration_performed",
        )
    ):
        raise RuntimeError("paired run execution boundary changed")
    return PublishedPairedRun(
        root=root,
        catalog_fingerprint=str(complete["pair_catalog_fingerprint"]),
        seed42_schedule_fingerprint=str(
            complete["seed42_schedule_fingerprint"]
        ),
        seed43_schedule_fingerprint=str(
            complete["seed43_schedule_fingerprint"]
        ),
        run_receipt_fingerprint=str(
            complete["run_receipt_fingerprint"]
        ),
        complete_fingerprint=str(complete["complete_fingerprint"]),
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
    geometry_config_path = _canonical_existing_file(
        args.geometry_config,
        name="geometry-safe config",
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
    paired_protocol_path = _canonical_existing_file(
        args.paired_protocol,
        name="paired-objective protocol",
    )
    output = _prepare_output(args.output)

    paired_protocol = _load_paired_protocol(paired_protocol_path)
    if file_sha256(geometry_config_path) != GEOMETRY_PROTOCOL_FILE_SHA256:
        raise RuntimeError("geometry config is not the exact frozen file")
    geometry_protocol = load_geometry_catalog_protocol(
        geometry_config_path
    )
    (
        upstream_geometry_catalog,
        upstream_p0_a1,
        upstream_eligible_view,
        upstream_geometry_complete,
    ) = _upstream_binding(
        paired_protocol,
        geometry_catalog_path=geometry_catalog_path,
        p0_a1_path=p0_a1_path,
        eligible_view_path=eligible_view_path,
        geometry_complete_path=geometry_complete_path,
        geometry_protocol=geometry_protocol,
    )

    manifest = load_and_validate_manifest(manifest_path)
    if manifest.dataset != geometry_protocol.dataset:
        raise RuntimeError("manifest dataset differs from geometry protocol")
    state_index = _strict_json(state_index_path, name="D_R state index")
    preprocess = _verify_geometry_input_binding(
        geometry_protocol,
        manifest_path,
        state_index_path,
        state_index,
    )
    immutable_files = {
        str(path): file_sha256(path)
        for path in (
            manifest_path,
            state_index_path,
            geometry_config_path,
            geometry_catalog_path,
            p0_a1_path,
            eligible_view_path,
            geometry_complete_path,
            paired_protocol_path,
        )
    }
    implementation_files = _implementation_binding()

    dataset = ManifestImageDataset(
        manifest,
        "D_R",
        preprocess,
        manifest_path=manifest_path,
    )
    bundle = load_d_r_cache_bundle(
        state_index_path,
        dataset,
        expected_base_fingerprint=(
            geometry_protocol.input_binding.base_fingerprint
        ),
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
    prepared = prepare_training_catalog(
        sources,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )
    geometry = build_geometry_safe_catalog(
        bundle,
        prepared,
        manifest,
        geometry_protocol,
    )
    reconstructed_geometry = _fingerprinted(geometry.canonical_payload())
    if reconstructed_geometry != upstream_geometry_catalog:
        raise RuntimeError(
            "reconstructed geometry catalog differs from authoritative P0-A1"
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
            "reconstructed P0-A1 receipt differs from the authority"
        )
    view = build_geometry_safe_p0_view(prepared, geometry)
    reconstructed_view = _reconstructed_eligible_view_receipt(
        geometry,
        view,
        str(upstream_p0_a1["eligible_catalog_fingerprint"]),
    )
    if reconstructed_view != upstream_eligible_view:
        raise RuntimeError(
            "reconstructed eligible view differs from the authority"
        )

    catalog = build_pair_catalog(
        prepared,
        geometry,
        manifest,
        paired_protocol_fingerprint=PAIRED_PROTOCOL_FINGERPRINT,
        match_config=bundle.match_config,
    )
    input_bindings: dict[str, object] = {
        "dataset": manifest.dataset,
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "paired_protocol_fingerprint": PAIRED_PROTOCOL_FINGERPRINT,
        "paired_protocol_file_sha256": file_sha256(
            paired_protocol_path
        ),
        "paired_protocol_repo_path": _repo_relative_path(
            paired_protocol_path,
            name="paired-objective protocol",
        ),
        "geometry_protocol_fingerprint": geometry_protocol.fingerprint,
        "geometry_protocol_file_sha256": file_sha256(
            geometry_config_path
        ),
        "geometry_protocol_repo_path": _repo_relative_path(
            geometry_config_path,
            name="geometry protocol",
        ),
        "geometry_catalog_fingerprint": geometry.catalog_fingerprint,
        "geometry_catalog_file_sha256": file_sha256(
            geometry_catalog_path
        ),
        "geometry_catalog_repo_path": _repo_relative_path(
            geometry_catalog_path,
            name="geometry catalog receipt",
        ),
        "geometry_catalog_frozen_repo_path": paired_protocol[
            "upstream_binding"
        ]["geometry_catalog_path"],
        "geometry_catalog_path_matches_frozen_binding": True,
        "p0_a1_receipt_fingerprint": upstream_p0_a1[
            "receipt_fingerprint"
        ],
        "p0_a1_file_sha256": file_sha256(p0_a1_path),
        "p0_a1_repo_path": _repo_relative_path(
            p0_a1_path,
            name="P0-A1 receipt",
        ),
        "p0_a1_frozen_repo_path": paired_protocol["upstream_binding"][
            "p0_a1_path"
        ],
        "p0_a1_path_matches_frozen_binding": True,
        "eligible_view_receipt_fingerprint": upstream_eligible_view[
            "receipt_fingerprint"
        ],
        "eligible_view_file_sha256": file_sha256(eligible_view_path),
        "eligible_view_repo_path": _repo_relative_path(
            eligible_view_path,
            name="eligible-view receipt",
        ),
        "eligible_view_frozen_repo_path": paired_protocol[
            "upstream_binding"
        ]["eligible_view_path"],
        "eligible_view_path_matches_frozen_binding": True,
        "geometry_complete_fingerprint": upstream_geometry_complete[
            "complete_fingerprint"
        ],
        "geometry_complete_file_sha256": file_sha256(
            geometry_complete_path
        ),
        "geometry_complete_repo_path": _repo_relative_path(
            geometry_complete_path,
            name="geometry COMPLETE",
        ),
        "manifest_fingerprint": manifest.fingerprint,
        "manifest_file_sha256": file_sha256(manifest_path),
        "state_index_fingerprint": bundle.state_index_fingerprint,
        "state_index_file_sha256": file_sha256(state_index_path),
        "base_fingerprint": bundle.base_fingerprint,
        "base_state_fingerprint": bundle.base_state_fingerprint,
        "state_fingerprint": bundle.state_fingerprint,
        "gt_fingerprint": bundle.gt_fingerprint,
        "source_catalog_fingerprint": geometry.source_catalog_fingerprint,
    }

    bundle.verify_unchanged()
    if any(
        file_sha256(Path(path)) != digest
        for path, digest in immutable_files.items()
    ):
        raise RuntimeError("a frozen paired-preflight input changed while loading")
    if _implementation_binding() != implementation_files:
        raise RuntimeError(
            "paired-preflight implementation changed while loading"
        )
    published = write_paired_run_artifact(
        catalog,
        output,
        input_bindings=input_bindings,
        implementation_files=implementation_files,
    )
    bundle.verify_unchanged()
    if any(
        file_sha256(Path(path)) != digest
        for path, digest in immutable_files.items()
    ):
        raise RuntimeError(
            "a frozen paired-preflight input changed during publication"
        )
    if _implementation_binding() != implementation_files:
        raise RuntimeError(
            "paired-preflight implementation changed during publication"
        )
    published.verify_unchanged()
    return {
        "output": str(published.root),
        "split": "D_R",
        "paired_protocol_fingerprint": PAIRED_PROTOCOL_FINGERPRINT,
        "pair_catalog_fingerprint": published.catalog_fingerprint,
        "counts": {
            "clean_positive": len(catalog.clean_positive),
            "component_null": len(catalog.component_null),
            "identity_null": len(catalog.identity_null),
            "exclusions": len(catalog.exclusions),
        },
        "seed42_schedule_fingerprint": (
            published.seed42_schedule_fingerprint
        ),
        "seed43_schedule_fingerprint": (
            published.seed43_schedule_fingerprint
        ),
        "complete_fingerprint": published.complete_fingerprint,
        "training_performed": False,
        "d_v_accessed": False,
        "d_t_accessed": False,
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
