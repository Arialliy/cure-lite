#!/usr/bin/env python3
"""Materialize the frozen NLCC-v12 role-quotient receipt exactly once."""

from __future__ import annotations

import argparse
from pathlib import Path

from cure_lite.cache.schema import canonical_json
from cure_lite.nlcc_role_quotient_audit import (
    build_nlcc_development_role_quotient_receipt,
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
    parser.add_argument(
        "--include-d4-diagnostic",
        action="store_true",
        help="Run the optional non-gating D4 diagnostic.",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    receipt = build_nlcc_development_role_quotient_receipt(
        max_examples=args.max_examples,
        max_records_per_label=args.max_records_per_label,
        include_d4_diagnostic=args.include_d4_diagnostic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(receipt) + "\n").encode("utf-8")
    with args.output.open("xb") as handle:
        handle.write(payload)

    decision = receipt["decision"]
    print(f"output={args.output}")
    print(f"receipt_fingerprint={receipt['receipt_fingerprint']}")
    print(
        "exact_conflicts="
        f"{receipt['exact_tensor']['conflict_key_count']}"
    )
    print(
        "signed_quotient_conflicts="
        f"{receipt['signed_amplitude_quotient']['conflict_key_count']}"
    )
    print(
        "role_quotient_conflicts="
        f"{receipt['role_quotient']['conflict_key_count']}"
    )
    print(f"hard_gate_pass={str(decision['hard_gate_pass']).lower()}")
    print(f"role_gate_pass={str(decision['role_gate_pass']).lower()}")
    print(
        "development_authorized="
        f"{str(decision['development_authorized']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
