#!/usr/bin/env python3
"""Create the frozen NLCC-v12 factorial conflict-attribution receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from cure_lite.cache.schema import canonical_json
from cure_lite.nlcc_role_conflict_attribution import (
    build_nlcc_development_role_conflict_attribution_receipt,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Create-only JSON receipt path.",
    )
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--max-records-per-label", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    receipt = (
        build_nlcc_development_role_conflict_attribution_receipt(
            max_examples=args.max_examples,
            max_records_per_label=args.max_records_per_label,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(receipt) + "\n").encode("utf-8")
    with args.output.open("xb") as handle:
        handle.write(payload)

    print(f"output={args.output}")
    print(f"receipt_fingerprint={receipt['receipt_fingerprint']}")
    for factor_id, result in receipt["factors"].items():
        print(
            f"{factor_id}_conflicts="
            f"{result['conflict_key_count']}"
        )
        print(
            f"{factor_id}_opposing_pairs="
            f"{result['opposing_record_pair_count']}"
        )
    print("training_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
