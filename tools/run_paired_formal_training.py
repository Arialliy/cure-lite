#!/usr/bin/env python3
"""Run one frozen no-resume paired CURE-Lite formal training attempt.

The command reconstructs only the authoritative ``D_R`` catalogs and the
already-preflighted 800 x 40 schedule.  It has no D_V/D_T, calibration,
inference, wave-decision, checkpoint, resume, overwrite, or horizon override
interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cure_lite.cache.schema import file_sha256  # noqa: E402
from cure_lite.experiment.paired_formal_controls import (  # noqa: E402
    build_paired_formal_control_provider,
    load_frozen_control_preflight_fingerprints,
)
from cure_lite.experiment.paired_formal_preflight import (  # noqa: E402
    load_paired_formal_preflight_artifact,
    validate_paired_formal_preflight_config,
)
from cure_lite.experiment.paired_formal_evaluation import (  # noqa: E402
    load_frozen_comparison_protocol,
)
from cure_lite.experiment.paired_formal_runner import (  # noqa: E402
    FORMAL_RUNNER_METHODS,
    FORMAL_RUNNER_SEEDS,
    PAIRED_DIFFERENCE_METHOD,
    PairedFormalRuntimeInputs,
    load_paired_formal_runner_config,
    run_paired_formal_attempt,
    validate_preflight_binding,
)
from cure_lite.experiment.paired_formal_schedule import (  # noqa: E402
    build_paired_formal_schedule,
)
from cure_lite.train.paired_pools import build_paired_schedule  # noqa: E402
from tools import run_paired_bounded_learnability as bounded_runner  # noqa: E402
from tools import run_paired_formal_preflight as preflight_runner  # noqa: E402
from tools import run_paired_preflight as pair_preflight_runner  # noqa: E402


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/paired_formal_runner_v1/config.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / _CONFIG_REPO_PATH,
    )
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=FORMAL_RUNNER_METHODS,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=FORMAL_RUNNER_SEEDS,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _canonical_file(path: Path, *, name: str) -> Path:
    if path.expanduser().is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    result = path.expanduser().resolve(strict=True)
    if not result.is_file() or result.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return result


def _canonical_directory(path: Path, *, name: str) -> Path:
    if path.expanduser().is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    result = path.expanduser().resolve(strict=True)
    if not result.is_dir() or result.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink directory")
    return result


def _implementation_binding() -> dict[str, str]:
    result = preflight_runner._implementation_binding()
    additional = (
        _ROOT / "tools" / "run_paired_formal_training.py",
        _ROOT / "cure_lite" / "paired_control_inputs.py",
        _ROOT / "cure_lite" / "paired_control_losses.py",
        _ROOT / "cure_lite" / "experiment" / "paired_artifacts.py",
        _ROOT / "cure_lite" / "experiment" / "paired_formal_controls.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "paired_formal_evaluation.py",
        _ROOT / "cure_lite" / "experiment" / "paired_formal_runner.py",
        _ROOT / "cure_lite" / "experiment" / "paired_formal_training.py",
        _ROOT / "cure_lite" / "train" / "paired_control_step.py",
    )
    result.update(
        {
            path.relative_to(_ROOT).as_posix(): file_sha256(path)
            for path in additional
        }
    )
    return dict(sorted(result.items()))


def _verify_bound_file(
    section: Mapping[str, object],
    *,
    path_key: str = "repo_path",
    sha_key: str = "file_sha256",
    name: str,
) -> Path:
    raw_path = section.get(path_key)
    expected_sha = section.get(sha_key)
    if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
        raise RuntimeError(f"{name} binding is malformed")
    path = _canonical_file(_ROOT / raw_path, name=name)
    if path.relative_to(_ROOT).as_posix() != raw_path:
        raise RuntimeError(f"{name} path differs from the freeze")
    if file_sha256(path) != expected_sha:
        raise RuntimeError(f"{name} SHA256 differs from the freeze")
    return path


def _device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must be an explicit torch CPU/CUDA device") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be an explicit torch CPU/CUDA device")
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("CUDA device must include an explicit index")
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise RuntimeError("requested CUDA device is unavailable")
    return device


def run(args: argparse.Namespace) -> dict[str, object]:
    expected_config = (_ROOT / _CONFIG_REPO_PATH).resolve(strict=True)
    config_path = _canonical_file(args.config, name="formal runner config")
    if config_path != expected_config:
        raise RuntimeError("formal runner config path differs from the freeze")
    protocol = load_paired_formal_runner_config(config_path)
    if _implementation_binding() != protocol.payload["implementation_binding"]:
        raise RuntimeError("formal runner implementation binding changed")

    preflight_root = _canonical_directory(
        args.preflight,
        name="formal schedule preflight",
    )
    preflight = load_paired_formal_preflight_artifact(preflight_root)
    validate_preflight_binding(protocol, preflight)

    reconstruction = protocol.payload["catalog_reconstruction_binding"]
    objective = protocol.payload["paired_objective_binding"]
    control = protocol.payload["control_preflight_binding"]
    evaluation = protocol.payload["evaluation_protocol_binding"]
    assert isinstance(reconstruction, Mapping)
    assert isinstance(objective, Mapping)
    assert isinstance(control, Mapping)
    assert isinstance(evaluation, Mapping)
    reconstruction_path = _verify_bound_file(
        reconstruction,
        name="catalog reconstruction config",
    )
    reconstruction_config = validate_paired_formal_preflight_config(
        pair_preflight_runner._strict_json(
            reconstruction_path,
            name="catalog reconstruction config",
        )
    )
    if (
        reconstruction_config["config_fingerprint"]
        != reconstruction["config_fingerprint"]
    ):
        raise RuntimeError("catalog reconstruction config identity changed")
    objective_path = _verify_bound_file(
        objective,
        name="paired objective protocol",
    )
    objective_payload = pair_preflight_runner._strict_json(
        objective_path,
        name="paired objective protocol",
    )
    if (
        objective_payload.get("receipt_fingerprint")
        != objective["receipt_fingerprint"]
    ):
        raise RuntimeError("paired objective protocol identity changed")
    evaluation_path = _verify_bound_file(
        evaluation,
        name="formal evaluation protocol",
    )
    comparison_protocol = load_frozen_comparison_protocol(evaluation_path)
    if comparison_protocol.comparison_protocol_fingerprint != (
        evaluation["comparison_protocol_fingerprint"]
    ):
        raise RuntimeError("formal evaluation protocol identity changed")

    pair_catalog, prepared, bundle, _ = bounded_runner._load_real_catalog(
        reconstruction_config
    )
    paired_schedule = build_paired_schedule(pair_catalog, seed=args.seed)
    schedule = build_paired_formal_schedule(paired_schedule, prepared)

    provider = None
    if args.method != PAIRED_DIFFERENCE_METHOD:
        control_root = _canonical_directory(
            _ROOT / str(control["repo_path"]),
            name="matched-control preflight",
        )
        if (
            file_sha256(control_root / "COMPLETE.json")
            != control["complete_file_sha256"]
        ):
            raise RuntimeError("matched-control preflight SHA256 changed")
        frozen_control = load_frozen_control_preflight_fingerprints(
            control_root,
            pair_catalog,
        )
        provider = build_paired_formal_control_provider(
            pair_catalog,
            prepared,
            frozen_control,
        )
        if provider.provider_fingerprint != control["provider_fingerprint"]:
            raise RuntimeError("formal control provider fingerprint changed")

    runtime = PairedFormalRuntimeInputs(
        bundle=bundle,
        pair_catalog=pair_catalog,
        prepared_catalog=prepared,
        schedule=schedule,
        control_provider=provider,
    )
    published = run_paired_formal_attempt(
        protocol,
        preflight,
        runtime,
        method=args.method,
        seed=args.seed,
        output_dir=args.output,
        device=_device(args.device),
    )
    return {
        "status": "PAIRED_FORMAL_TRAINING_COMPLETE",
        "output": str(published.root),
        "dataset": "IRSTD-1K",
        "training_split": "D_R",
        "method": published.method,
        "seed": published.seed,
        "formal_schedule_fingerprint": (
            published.formal_schedule_fingerprint
        ),
        "initial_decoder_fingerprint": (
            published.initial_decoder_fingerprint
        ),
        "final_decoder_fingerprint": published.final_decoder_fingerprint,
        "paired_artifact_fingerprint": (
            published.paired_artifact_fingerprint
        ),
        "complete_fingerprint": published.complete_fingerprint,
        "optimizer_updates": 32_000,
        "checkpoint_written": False,
        "resume_used": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "calibration_performed": False,
        "inference_performed": False,
        "wave_decision_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as error:
        print(
            f"PAIRED_FORMAL_TRAINING_ERROR: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
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
