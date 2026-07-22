#!/usr/bin/env python3
"""Export generic D_R/D_V Base caches from a completed reference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.data import ManifestImageDataset  # noqa: E402
from cure_lite.experiment.cache_pipeline import cache_manifest_split  # noqa: E402
from cure_lite.reference_base import (  # noqa: E402
    load_reference_base_adapter,
    load_reference_base_run,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402


CACHE_RUN_SCHEMA = "cure-lite-reference-base-cache-run-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"reference-base cache output already exists: {output}")
    output = output.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = load_and_validate_manifest(manifest_path)
    loaded = load_reference_base_run(args.reference_base_run, device="cpu")
    if (
        loaded.manifest_fingerprint != manifest.fingerprint
        or loaded.manifest_file_sha256 != file_sha256(manifest_path)
    ):
        raise ValueError("reference-base run differs from the requested manifest")
    adapter = load_reference_base_adapter(
        args.reference_base_run,
        device=args.device,
    )
    preprocess = loaded.config.preprocess
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
    d_r = cache_manifest_split(adapter, d_r_dataset, "D_R", output / "D_R")
    d_v = cache_manifest_split(adapter, d_v_dataset, "D_V", output / "D_V")
    summary: dict[str, object] = {
        "schema_version": CACHE_RUN_SCHEMA,
        "dataset": manifest.dataset,
        "manifest_fingerprint": manifest.fingerprint,
        "manifest_file_sha256": file_sha256(manifest_path),
        "base_fingerprint": loaded.base_fingerprint,
        "reference_base_run_fingerprint": json.loads(
            (loaded.root / "COMPLETE.json").read_text(encoding="utf-8")
        )["run_fingerprint"],
        "d_r_index_fingerprint": d_r["index_fingerprint"],
        "d_r_index_sha256": file_sha256(output / "D_R" / "index.json"),
        "d_v_index_fingerprint": d_v["index_fingerprint"],
        "d_v_index_sha256": file_sha256(output / "D_V" / "index.json"),
    }
    summary["cache_run_fingerprint"] = stable_fingerprint(summary)
    _write_summary(output / "COMPLETE.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
