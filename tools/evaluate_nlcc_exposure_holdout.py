#!/usr/bin/env python3
"""Canonical NLCC-v12 unique dataset-free exposure-holdout entrypoint.

Holdout uses development PASS artifacts only for authorization.  It always
starts from a new seed-42 decoder and a fresh empty Adam state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.nlcc_dataset_free_runner import (  # noqa: E402
    preflight_profile,
    run_canonical_profile,
)
from cure_lite.nlcc_dataset_free_runner_config import (  # noqa: E402
    holdout_runner_config,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate frozen source/input structure without claiming or training",
    )
    args = parser.parse_args(argv)
    config = holdout_runner_config()
    if args.preflight:
        print(json.dumps(preflight_profile(config), sort_keys=True))
        return 0
    try:
        sealed, status = run_canonical_profile(config)
    except BaseException as error:
        print(
            json.dumps(
                {
                    "profile_id": config.profile.profile_id,
                    "status": "PRE_ATTEMPT_FAILURE",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "artifact_directory_created": False,
                },
                sort_keys=True,
            )
        )
        return 3
    print(json.dumps(sealed, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
