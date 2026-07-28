#!/usr/bin/env python3
"""Validate, execute, or externally finalize the fixed PAET D_V recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

# Establish the frozen GPU mapping before importing the original torch runner.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite_eval_v2.formal_d_v_runner_v2 import (  # noqa: E402
    finalize_paet_formal_d_v_evidence_binding_v2,
    run_paet_formal_d_v_once_v2,
    validate_paet_formal_d_v_create_only_v2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-create-only",
        action="store_true",
        help=(
            "validate both source closures, the erratum, the original "
            "Formal800 loader, and the fixed plan without opening D_V"
        ),
    )
    mode.add_argument(
        "--run-once",
        action="store_true",
        help="execute the one original fixed D_V evaluation exactly once",
    )
    mode.add_argument(
        "--finalize-evidence-binding",
        action="store_true",
        help=(
            "after a completed D_V publication, create or verify only the "
            "external v2 evidence-binding receipt without rerunning D_V"
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_create_only:
        payload = validate_paet_formal_d_v_create_only_v2()
    elif args.run_once:
        payload = run_paet_formal_d_v_once_v2()
    else:
        payload = finalize_paet_formal_d_v_evidence_binding_v2()
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
