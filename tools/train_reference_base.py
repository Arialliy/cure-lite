#!/usr/bin/env python3
"""Train the project-owned Stage-A reference Base on D_B only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.reference_base import (  # noqa: E402
    ReferenceBaseTrainingConfig,
    train_reference_base,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _json_object(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("reference-base config must be a regular file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"reference-base config repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"reference-base config contains non-finite {value}")

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    if not isinstance(payload, Mapping):
        raise ValueError("reference-base config root must be an object")
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = load_and_validate_manifest(manifest_path)
    config = ReferenceBaseTrainingConfig.from_mapping(_json_object(args.config))
    loaded = train_reference_base(
        manifest,
        manifest_path,
        config,
        args.output,
    )
    print(
        json.dumps(
            {
                "dataset": config.dataset,
                "epochs_completed": config.epochs,
                "best_epoch": loaded.best_epoch,
                "best_select_global_miou": loaded.best_select_miou,
                "best_select_loss": loaded.best_select_loss,
                "base_fingerprint": loaded.base_fingerprint,
                "checkpoint_sha256": loaded.checkpoint_sha256,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
