#!/usr/bin/env python3
"""Run one fresh 800 x 40 PFCR CURE-Lite training attempt on D_R."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cure_lite.cache.schema import file_sha256  # noqa: E402
from cure_lite.data import ManifestImageDataset, PreprocessConfig  # noqa: E402
from cure_lite.experiment.cache_pipeline import (  # noqa: E402
    load_d_r_cache_bundle,
)
from cure_lite.experiment.geometry_catalog_protocol import (  # noqa: E402
    load_geometry_catalog_protocol,
)
from cure_lite.experiment.phase_resolved_real_formal_runner import (  # noqa: E402
    load_pfcr_development_authorization,
    load_pfcr_real_preflight_authorization,
    run_pfcr_real_formal_attempt,
)
from cure_lite.experiment.phase_resolved_real_training import (  # noqa: E402
    PFCRRealFormalTrainingConfig,
)
from cure_lite.phase_resolved_real_cache import (  # noqa: E402
    adapt_pfcr_d_r_cache,
)
from cure_lite.phase_resolved_real_states import (  # noqa: E402
    build_pfcr_real_state_catalog,
    load_pfcr_lineage_allowlist,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402


_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--p0-a1", type=Path, required=True)
    parser.add_argument(
        "--development-seed42",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--development-seed43",
        type=Path,
        required=True,
    )
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        choices=(42, 43),
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_file(path: Path, *, name: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    result = raw.resolve(strict=True)
    if not result.is_file() or result.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return result


def _device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError(
            "device must be an explicit CPU or CUDA device"
        ) from error
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU device cannot include an index")
        return device
    if device.type != "cuda" or device.index is None:
        raise ValueError("CUDA device must include an explicit index")
    if (
        not torch.cuda.is_available()
        or device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def _implementation_binding() -> dict[str, str]:
    paths = (
        _ROOT / "cure_lite/phase_resolved_feature_coverage_relation.py",
        _ROOT / "cure_lite/phase_resolved_relation_decoder.py",
        _ROOT / "cure_lite/phase_resolved_relation_training.py",
        _ROOT / "cure_lite/phase_resolved_real_cache.py",
        _ROOT / "cure_lite/phase_resolved_real_states.py",
        _ROOT / "cure_lite/sampling.py",
        _ROOT / "cure_lite/train/pools.py",
        _ROOT / "cure_lite/train/phase_resolved_relation_step.py",
        _ROOT / "cure_lite/experiment/phase_resolved_real_training.py",
        _ROOT
        / "cure_lite/experiment/phase_resolved_real_artifacts.py",
        _ROOT
        / "cure_lite/experiment/phase_resolved_real_formal_runner.py",
        _ROOT / "tools/train_phase_resolved_relation_real.py",
    )
    return {
        path.relative_to(_ROOT).as_posix(): file_sha256(path)
        for path in paths
    }


def _input_binding(
    paths: Mapping[str, Path],
) -> dict[str, str]:
    return {
        path.relative_to(_ROOT).as_posix(): file_sha256(path)
        for path in sorted(paths.values())
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "manifest": _canonical_file(
            args.manifest,
            name="manifest",
        ),
        "state_index": _canonical_file(
            args.state_index,
            name="state index",
        ),
        "geometry_config": _canonical_file(
            args.geometry_config,
            name="geometry config",
        ),
        "p0_a1": _canonical_file(
            args.p0_a1,
            name="P0-A1 receipt",
        ),
        "development_seed42": _canonical_file(
            args.development_seed42,
            name="Development seed42 result",
        ),
        "development_seed43": _canonical_file(
            args.development_seed43,
            name="Development seed43 result",
        ),
    }
    device = _device(args.device)
    authorization = load_pfcr_real_preflight_authorization(
        args.preflight,
        expected_seed=args.seed,
    )
    development_authorization = (
        load_pfcr_development_authorization(
            paths["development_seed42"],
            paths["development_seed43"],
        )
    )
    manifest = load_and_validate_manifest(paths["manifest"])
    with paths["state_index"].open("r", encoding="utf-8") as handle:
        state_index = json.load(handle)
    if not isinstance(state_index, dict):
        raise ValueError("state index must contain one JSON object")
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

    def progress(row: Mapping[str, object]) -> None:
        epoch = int(row["epoch"])
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == 799:
            metrics = row["metrics"]
            assert isinstance(metrics, Mapping)
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "epochs": 800,
                        "updates_completed": (
                            row["optimizer_updates_completed"]
                        ),
                        "mean_total_loss": (
                            metrics["mean_total_loss"]
                        ),
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )

    published = run_pfcr_real_formal_attempt(
        cache,
        catalog,
        PFCRRealFormalTrainingConfig(seed=args.seed),
        authorization,
        development_authorization,
        output_dir=args.output,
        device=device,
        binding_root=_ROOT,
        input_binding=_input_binding(paths),
        implementation_binding=_implementation_binding(),
        epoch_callback=progress,
    )
    return {
        "status": "PFCR_REAL_FORMAL_TRAINING_COMPLETE",
        "model": "CURE-Lite",
        "dataset": cache.contract.dataset,
        "training_split": "D_R",
        "seed": published.seed,
        "output": str(published.root),
        "artifact_fingerprint": (
            published.artifact.artifact_fingerprint
        ),
        "final_model_fingerprint": (
            published.artifact.decoder_state_fingerprint
        ),
        "optimizer_updates": (
            published.artifact.execution_ledger.optimizer_updates
        ),
        "D_V_read": False,
        "performance_success_claimed": False,
        "D_V_evaluation_authorized": True,
        "full_CURE_authorized": False,
    }


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
