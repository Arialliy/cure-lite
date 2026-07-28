#!/usr/bin/env python3
"""Validate or execute the sole fixed PAET-BFA Formal800 D_V evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

# Establish the fixed physical-GPU/environment mapping before importing
# torch through the experiment package.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.experiment.coverage_state_paet_formal_d_v_runner import (  # noqa: E402
    run_paet_formal_d_v_once,
    validate_paet_formal_d_v_create_only,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-create-only",
        action="store_true",
        help="validate the fixed plan without opening D_V or creating output",
    )
    mode.add_argument(
        "--run-once",
        action="store_true",
        help="claim and execute the one fixed D_V evaluation exactly once",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = (
        validate_paet_formal_d_v_create_only()
        if args.validate_create_only
        else run_paet_formal_d_v_once()
    )
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
