#!/usr/bin/env python3
"""Create one CURE-Lite relation-decoder Development result."""

from __future__ import annotations

import argparse
from pathlib import Path

from cure_lite.cache.schema import canonical_json
from cure_lite.phase_resolved_relation_training import (
    PhaseResolvedRelationTrainingConfig,
    run_phase_resolved_relation_development,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Create-only result JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    result = run_phase_resolved_relation_development(
        PhaseResolvedRelationTrainingConfig(seed=args.seed)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(result) + "\n").encode("utf-8")
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(f"output={args.output}")
    print(f"result_fingerprint={result['result_fingerprint']}")
    metrics = result["final_metrics"]
    print(
        "mismatch_pixels="
        f"{metrics['lossless_threshold_mismatch_pixel_count']}"
    )
    print(
        "positive_probability_min="
        f"{metrics['positive_probability_min']:.9f}"
    )
    print(
        "negative_probability_max="
        f"{metrics['negative_probability_max']:.9f}"
    )
    print(
        "development_learnability_pass="
        f"{str(result['decision']['development_learnability_pass']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
