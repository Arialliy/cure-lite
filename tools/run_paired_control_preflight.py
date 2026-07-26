#!/usr/bin/env python3
"""Publish the real D_R matched-control preflight for frozen CURE-Lite.

This create-only runner reconstructs the same real pair catalog as the sealed
paired preflight, verifies the frozen protocol/catalog identities, and emits
only tensor-free control contracts, the real-shape DCT basis description, and
the deterministic target-permutation plan.  It performs no model forward,
training, calibration, inference, or access outside ``D_R``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.data import ManifestImageDataset  # noqa: E402
from cure_lite.experiment.cache_pipeline import (  # noqa: E402
    load_d_r_cache_bundle,
)
from cure_lite.experiment.geometry_catalog_protocol import (  # noqa: E402
    load_geometry_catalog_protocol,
)
from cure_lite.experiment.geometry_safe_catalog import (  # noqa: E402
    build_geometry_safe_catalog,
    build_geometry_safe_p0_view,
    build_p0_a1_receipt,
)
from cure_lite.experiment.paired_catalog import build_pair_catalog  # noqa: E402
from cure_lite.experiment.paired_control_preflight import (  # noqa: E402
    CONTROL_PREFLIGHT_CATALOG_FINGERPRINT,
    CONTROL_PREFLIGHT_CLEAN_PAIR_COUNT,
    CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT,
    REQUIRED_CONTROL_SOURCE_PATHS,
    write_control_preflight_artifact,
)
from cure_lite.experiment.training_pipeline import (  # noqa: E402
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402
from tools.run_paired_preflight import (  # noqa: E402
    GEOMETRY_PROTOCOL_FILE_SHA256,
    _canonical_existing_file,
    _fingerprinted,
    _implementation_binding,
    _load_paired_protocol,
    _prepare_output,
    _reconstructed_eligible_view_receipt,
    _repo_relative_path,
    _strict_json,
    _upstream_binding,
    _verify_geometry_input_binding,
    load_paired_run_artifact,
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
    parser.add_argument("--p0-a1-receipt", type=Path, required=True)
    parser.add_argument("--eligible-view-receipt", type=Path, required=True)
    parser.add_argument("--geometry-complete", type=Path, required=True)
    parser.add_argument("--paired-protocol", type=Path, required=True)
    parser.add_argument(
        "--paired-preflight-complete",
        type=Path,
        required=True,
        help="COMPLETE.json from the sealed real paired preflight authority",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _control_source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in REQUIRED_CONTROL_SOURCE_PATHS:
        path = (_ROOT / relative).resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"control source must be a regular file: {relative}"
            )
        if path.relative_to(_ROOT).as_posix() != relative:
            raise RuntimeError(f"control source path is not canonical: {relative}")
        hashes[relative] = file_sha256(path)
    return dict(sorted(hashes.items()))


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
    paired_preflight_complete_path = _canonical_existing_file(
        args.paired_preflight_complete,
        name="paired-preflight COMPLETE",
    )
    if paired_preflight_complete_path.name != "COMPLETE.json":
        raise RuntimeError(
            "paired-preflight authority must be its root COMPLETE.json"
        )
    output = _prepare_output(args.output)

    paired_authority = load_paired_run_artifact(
        paired_preflight_complete_path.parent
    )
    paired_authority_complete = _strict_json(
        paired_preflight_complete_path,
        name="paired-preflight COMPLETE",
    )
    if (
        paired_authority.catalog_fingerprint
        != CONTROL_PREFLIGHT_CATALOG_FINGERPRINT
        or paired_authority_complete.get("pair_catalog_fingerprint")
        != CONTROL_PREFLIGHT_CATALOG_FINGERPRINT
        or paired_authority_complete.get("paired_protocol_fingerprint")
        != CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT
    ):
        raise RuntimeError(
            "paired-preflight authority differs from the matched-control freeze"
        )

    paired_protocol = _load_paired_protocol(paired_protocol_path)
    if (
        paired_protocol["receipt_fingerprint"]
        != CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT
    ):
        raise RuntimeError("paired-objective protocol differs from the freeze")
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
            paired_preflight_complete_path,
        )
    }
    strict_loader_source_hashes = _implementation_binding()
    control_source_hashes = _control_source_hashes()

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
        paired_protocol_fingerprint=(
            CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT
        ),
        match_config=bundle.match_config,
    )
    if (
        catalog.catalog_fingerprint
        != CONTROL_PREFLIGHT_CATALOG_FINGERPRINT
        or len(catalog.clean_positive) != CONTROL_PREFLIGHT_CLEAN_PAIR_COUNT
    ):
        raise RuntimeError(
            "reconstructed pair catalog differs from the matched-control freeze"
        )

    input_bindings: dict[str, object] = {
        "dataset": manifest.dataset,
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "paired_protocol_fingerprint": (
            CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT
        ),
        "paired_protocol_file_sha256": file_sha256(
            paired_protocol_path
        ),
        "paired_protocol_repo_path": _repo_relative_path(
            paired_protocol_path,
            name="paired-objective protocol",
        ),
        "pair_catalog_fingerprint": catalog.catalog_fingerprint,
        "paired_preflight_authority_repo_path": _repo_relative_path(
            paired_preflight_complete_path,
            name="paired-preflight COMPLETE",
        ),
        "upstream_paired_preflight_complete_fingerprint": (
            paired_authority.complete_fingerprint
        ),
        "upstream_paired_preflight_complete_sha256": file_sha256(
            paired_preflight_complete_path
        ),
        "upstream_paired_preflight_run_receipt_fingerprint": (
            paired_authority.run_receipt_fingerprint
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
        "p0_a1_receipt_fingerprint": upstream_p0_a1[
            "receipt_fingerprint"
        ],
        "p0_a1_file_sha256": file_sha256(p0_a1_path),
        "p0_a1_repo_path": _repo_relative_path(
            p0_a1_path,
            name="P0-A1 receipt",
        ),
        "eligible_view_receipt_fingerprint": upstream_eligible_view[
            "receipt_fingerprint"
        ],
        "eligible_view_file_sha256": file_sha256(eligible_view_path),
        "eligible_view_repo_path": _repo_relative_path(
            eligible_view_path,
            name="eligible-view receipt",
        ),
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
        "strict_loader_source_hashes_fingerprint": stable_fingerprint(
            strict_loader_source_hashes
        ),
    }

    bundle.verify_unchanged()
    paired_authority.verify_unchanged()
    if any(
        file_sha256(Path(path)) != digest
        for path, digest in immutable_files.items()
    ):
        raise RuntimeError("a frozen control-preflight input changed while loading")
    if _implementation_binding() != strict_loader_source_hashes:
        raise RuntimeError("the strict D_R loader changed while loading")
    if _control_source_hashes() != control_source_hashes:
        raise RuntimeError("a matched-control source changed while loading")

    published = write_control_preflight_artifact(
        catalog,
        output,
        input_bindings=input_bindings,
        control_source_hashes=control_source_hashes,
    )

    bundle.verify_unchanged()
    paired_authority.verify_unchanged()
    if any(
        file_sha256(Path(path)) != digest
        for path, digest in immutable_files.items()
    ):
        raise RuntimeError(
            "a frozen control-preflight input changed during publication"
        )
    if _implementation_binding() != strict_loader_source_hashes:
        raise RuntimeError("the strict D_R loader changed during publication")
    if _control_source_hashes() != control_source_hashes:
        raise RuntimeError(
            "a matched-control source changed during publication"
        )
    published.verify_unchanged()
    return {
        "output": str(published.root),
        "split": "D_R",
        "paired_protocol_fingerprint": published.protocol_fingerprint,
        "pair_catalog_fingerprint": published.catalog_fingerprint,
        "clean_pair_count": CONTROL_PREFLIGHT_CLEAN_PAIR_COUNT,
        "status": published.status,
        "dct_basis_fingerprint": published.dct_basis_fingerprint,
        "permutation_fingerprint": published.permutation_fingerprint,
        "complete_fingerprint": published.complete_fingerprint,
        "training_performed": False,
        "d_v_accessed": False,
        "d_t_accessed": False,
        "model_modified": False,
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
