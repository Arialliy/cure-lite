#!/usr/bin/env python3
"""Create the CURE-Lite relation-state v2 preflight receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from cure_lite.cache.schema import canonical_json
from cure_lite.phase_resolved_relation_preflight import (
    build_phase_resolved_relation_preflight_receipt,
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
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    receipt = build_phase_resolved_relation_preflight_receipt(
        max_examples=args.max_examples
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(receipt) + "\n").encode("utf-8")
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(f"output={args.output}")
    print(f"receipt_fingerprint={receipt['receipt_fingerprint']}")
    print(
        "input_contract_v2_pass="
        f"{str(receipt['decision']['input_contract_v2_pass']).lower()}"
    )
    print(
        "relation_role_conflicts="
        f"{receipt['relation_role_identifiability']['conflict_key_count']}"
    )
    print(
        "analytic_mismatch_pixels="
        f"{receipt['analytic_reference']['mismatch_pixel_count']}"
    )
    print("full_decoder_training_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
