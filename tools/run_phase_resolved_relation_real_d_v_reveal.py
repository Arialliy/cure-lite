#!/usr/bin/env python3
"""Run the frozen one-shot PFCR CURE-Lite formal D_V reveal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.experiment.phase_resolved_real_d_v_reveal import (  # noqa: E402
    run_pfcr_real_d_v_reveal,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        published = run_pfcr_real_d_v_reveal(args.config)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            published.success_summary(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
