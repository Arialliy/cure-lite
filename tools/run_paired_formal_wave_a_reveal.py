#!/usr/bin/env python3
"""Run the frozen one-shot paired CURE-Lite Wave-A reveal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.experiment.paired_formal_wave_reveal import (  # noqa: E402
    run_wave_a_reveal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    published = run_wave_a_reveal(args.config)
    print(
        json.dumps(
            published.success_summary(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
