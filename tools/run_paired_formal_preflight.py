#!/usr/bin/env python3
"""Publish the frozen D_R-only paired formal schedules for seeds 42/43.

This entrypoint reconstructs the already-authoritative real pair catalog and
prepared factual catalog, builds both complete 800 x 40 schedules, and emits
only tensor-free schedule receipts.  It has no training, model-forward,
calibration, inference, resume, overwrite, or D_V/D_T interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256  # noqa: E402
from cure_lite.experiment.paired_formal_preflight import (  # noqa: E402
    FORMAL_PREFLIGHT_SEEDS,
    load_paired_formal_preflight_artifact,
    validate_paired_formal_preflight_config,
    write_paired_formal_preflight_artifact,
)
from cure_lite.experiment.paired_formal_schedule import (  # noqa: E402
    build_paired_formal_schedule,
)
from cure_lite.train.paired_pools import build_paired_schedule  # noqa: E402
from tools import run_paired_bounded_learnability as bounded_runner  # noqa: E402
from tools import run_paired_preflight as pair_preflight_runner  # noqa: E402


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/paired_formal_preflight_v1/config.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / _CONFIG_REPO_PATH,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _canonical_file(path: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"{name} must be a regular non-symlink file")
    return resolved


def _load_config(path: Path) -> dict[str, object]:
    expected = (_ROOT / _CONFIG_REPO_PATH).resolve(strict=True)
    if path != expected:
        raise RuntimeError("formal preflight config path differs from the freeze")
    payload = pair_preflight_runner._strict_json(
        path,
        name="paired formal preflight config",
    )
    return validate_paired_formal_preflight_config(payload)


def _implementation_binding() -> dict[str, str]:
    """Return all code files used directly or by real-catalog reconstruction."""

    result = bounded_runner._implementation_binding()
    additional = (
        _ROOT / "tools" / "run_paired_formal_preflight.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "paired_formal_preflight.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "paired_formal_schedule.py",
    )
    result.update(
        {
            path.relative_to(_ROOT).as_posix(): file_sha256(path)
            for path in additional
        }
    )
    return dict(sorted(result.items()))


def _verify_formal_seed_configs(
    config: Mapping[str, object],
) -> dict[str, str]:
    binding = config.get("input_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("formal preflight input binding is malformed")
    actual: dict[str, str] = {}
    common_training: dict[str, object] | None = None
    for seed in FORMAL_PREFLIGHT_SEEDS:
        prefix = f"formal_seed{seed}_config"
        path_value = binding.get(f"{prefix}_path")
        expected_sha = binding.get(f"{prefix}_file_sha256")
        if not isinstance(path_value, str) or not isinstance(
            expected_sha,
            str,
        ):
            raise RuntimeError(f"{prefix} binding is malformed")
        path = _canonical_file(_ROOT / path_value, name=prefix)
        if file_sha256(path) != expected_sha:
            raise RuntimeError(f"frozen {prefix} changed")
        payload = pair_preflight_runner._strict_json(
            path,
            name=prefix,
        )
        training = payload.get("training")
        if not isinstance(training, Mapping):
            raise RuntimeError(f"{prefix} training contract is malformed")
        exact = {
            "global_seed": seed,
            "epochs": 800,
            "steps_per_epoch": 40,
            "factual_miss_batch": 4,
            "factual_no_miss_batch": 4,
            "synthetic_batch": 4,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        }
        if any(training.get(field) != value for field, value in exact.items()):
            raise RuntimeError(f"{prefix} formal budget or optimizer changed")
        comparable = {
            key: value
            for key, value in training.items()
            if key != "global_seed"
        }
        if common_training is None:
            common_training = comparable
        elif comparable != common_training:
            raise RuntimeError(
                "seed-42 and seed-43 formal training contracts differ"
            )
        actual[path.relative_to(_ROOT).as_posix()] = expected_sha
    return actual


def _normalize_immutable_inputs(
    values: Mapping[str, str],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_path, digest in values.items():
        path = Path(raw_path).resolve(strict=True)
        try:
            relative = path.relative_to(_ROOT).as_posix()
        except ValueError as error:
            raise RuntimeError(
                "formal preflight input escaped the repository root"
            ) from error
        normalized[relative] = digest
    return dict(sorted(normalized.items()))


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = _canonical_file(args.config, name="formal preflight config")
    config = _load_config(config_path)
    output = args.output.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite formal preflight output {output}"
        )
    implementation = _implementation_binding()
    if implementation != config["implementation_binding"]:
        raise RuntimeError("formal preflight implementation code SHA changed")
    formal_config_inputs = _verify_formal_seed_configs(config)

    pair_catalog, prepared, bundle, immutable = (
        bounded_runner._load_real_catalog(config)
    )
    if pair_catalog.split != "D_R":
        raise RuntimeError("formal schedule preflight may use only D_R")
    if pair_catalog.dataset != "IRSTD-1K":
        raise RuntimeError("formal schedule preflight dataset changed")
    schedules = {}
    for seed in FORMAL_PREFLIGHT_SEEDS:
        paired = build_paired_schedule(pair_catalog, seed=seed)
        schedules[seed] = build_paired_formal_schedule(paired, prepared)
    bundle.verify_unchanged()

    input_files = _normalize_immutable_inputs(immutable)
    input_files.update(formal_config_inputs)
    input_files = dict(sorted(input_files.items()))
    published = write_paired_formal_preflight_artifact(
        schedules,
        config=config,
        config_file_sha256=file_sha256(config_path),
        input_file_sha256=input_files,
        implementation_file_sha256=implementation,
        output_dir=output,
    )
    published.verify_unchanged()
    loaded = load_paired_formal_preflight_artifact(output)
    return {
        "status": "PAIRED_FORMAL_SCHEDULE_PREFLIGHT_COMPLETE",
        "output": str(loaded.root),
        "split": "D_R",
        "seeds": list(FORMAL_PREFLIGHT_SEEDS),
        "methods": list(config["methods"]),
        "prepared_catalog_fingerprint": (
            loaded.prepared_catalog_fingerprint
        ),
        "pair_catalog_fingerprint": loaded.pair_catalog_fingerprint,
        "seed42_formal_schedule_fingerprint": (
            loaded.seed42_formal_schedule_fingerprint
        ),
        "seed43_formal_schedule_fingerprint": (
            loaded.seed43_formal_schedule_fingerprint
        ),
        "method_bindings_fingerprint": (
            loaded.method_bindings_fingerprint
        ),
        "complete_fingerprint": loaded.complete_fingerprint,
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "formal_training_authorized_by_this_artifact": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as error:
        print(
            f"PAIRED_FORMAL_SCHEDULE_PREFLIGHT_ERROR: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    import json

    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
