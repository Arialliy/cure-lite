#!/usr/bin/env python3
"""Create once or validate the independent PAET evaluation-v3 closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite_eval_v3.evaluation_source_closure import (  # noqa: E402
    build_evaluation_v3_source_closure,
    verify_evaluation_v3_source_closure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-once", action="store_true")
    mode.add_argument("--validate", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = (
        build_evaluation_v3_source_closure()
        if args.create_once
        else verify_evaluation_v3_source_closure()
    )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
