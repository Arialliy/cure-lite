#!/usr/bin/env python3
"""Create one D_R-only PFCR real-execution preflight artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.data import ManifestImageDataset, PreprocessConfig
from cure_lite.experiment.cache_pipeline import load_d_r_cache_bundle
from cure_lite.experiment.geometry_catalog_protocol import (
    load_geometry_catalog_protocol,
)
from cure_lite.experiment.phase_resolved_real_training import (
    PFCRRealPreflightConfig,
    run_pfcr_real_preflight,
)
from cure_lite.phase_resolved_real_cache import adapt_pfcr_d_r_cache
from cure_lite.phase_resolved_real_states import (
    build_pfcr_real_state_catalog,
    load_pfcr_lineage_allowlist,
)
from cure_lite.splits import load_and_validate_manifest


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--p0-a1", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "manifest": args.manifest.expanduser().resolve(strict=True),
        "state_index": (
            args.state_index.expanduser().resolve(strict=True)
        ),
        "geometry_config": (
            args.geometry_config.expanduser().resolve(strict=True)
        ),
        "p0_a1": args.p0_a1.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=False)
    started = {
        "schema_version": "cure-lite-pfcr-real-preflight-start-v1",
        "seed": args.seed,
        "device": args.device,
        "input_files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        },
        "continuation_supported": False,
        "incomplete_attempt_may_be_reused": False,
    }
    started["receipt_fingerprint"] = stable_fingerprint(started)
    _write_new_json(output / "STARTED.json", started)

    manifest = load_and_validate_manifest(paths["manifest"])
    state_index = _load_json(paths["state_index"])
    preprocess = PreprocessConfig.from_fingerprint_payload(
        state_index["preprocessing"]
    )
    dataset = ManifestImageDataset(
        manifest,
        "D_R",
        preprocess,
        manifest_path=paths["manifest"],
    )
    geometry_protocol = load_geometry_catalog_protocol(
        paths["geometry_config"]
    )
    bundle = load_d_r_cache_bundle(
        paths["state_index"],
        dataset,
        expected_base_fingerprint=(
            geometry_protocol.input_binding.base_fingerprint
        ),
    )
    cache = adapt_pfcr_d_r_cache(bundle)
    allowlist = load_pfcr_lineage_allowlist(paths["p0_a1"])
    catalog = build_pfcr_real_state_catalog(cache, allowlist)
    result = run_pfcr_real_preflight(
        cache,
        catalog,
        PFCRRealPreflightConfig(seed=args.seed),
        device=args.device,
    )
    _write_new_json(output / "result.json", result)
    complete = {
        "schema_version": "cure-lite-pfcr-real-preflight-complete-v1",
        "result_file": "result.json",
        "result_file_sha256": file_sha256(output / "result.json"),
        "result_fingerprint": result["result_fingerprint"],
        "decision": result["decision"],
        "started_receipt_fingerprint": started["receipt_fingerprint"],
    }
    complete["complete_fingerprint"] = stable_fingerprint(complete)
    _write_new_json(output / "COMPLETE.json", complete)
    return complete


def main() -> int:
    args = _parser().parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
