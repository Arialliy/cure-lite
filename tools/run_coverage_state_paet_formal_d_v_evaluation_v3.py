#!/usr/bin/env python3
"""Validate, execute, or externally finalize PAET evaluation-v3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

# Establish the frozen GPU mapping before importing the torch-based runner.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite_eval_v3.formal_d_v_runner_v3 import (  # noqa: E402
    finalize_paet_formal_d_v_evidence_binding_v3,
    run_paet_formal_d_v_once_v3,
    validate_paet_formal_d_v_create_only_v3,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-create-only",
        action="store_true",
        help=(
            "verify the full closure/amendment/failure chain and the new "
            "fixed plan without opening D_V or D_T"
        ),
    )
    mode.add_argument(
        "--run-once",
        action="store_true",
        help="execute the distinct evaluation-v3 D_V attempt exactly once",
    )
    mode.add_argument(
        "--finalize-evidence-binding",
        action="store_true",
        help=(
            "after a completed v3 D_V publication, create or verify only "
            "its external evidence binding without rerunning D_V"
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_create_only:
        payload = validate_paet_formal_d_v_create_only_v3()
    elif args.run_once:
        payload = run_paet_formal_d_v_once_v3()
    else:
        payload = finalize_paet_formal_d_v_evidence_binding_v3()
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
