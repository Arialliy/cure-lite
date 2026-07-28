#!/usr/bin/env python3
"""Create once or validate the sealed PAET-BFA Formal800 source closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.coverage_state_paet_formal_source_closure import (
    build_coverage_state_paet_formal_source_closure,
    verify_coverage_state_paet_formal_source_closure,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-once", action="store_true")
    mode.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = (
        build_coverage_state_paet_formal_source_closure()
        if args.create_once
        else verify_coverage_state_paet_formal_source_closure()
    )
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
