#!/usr/bin/env python3
"""Preflight, execute, or compare the late tiny-target capacity audit.

No command accepts a dataset, split, device, seed, checkpoint, weight, training
or inference argument.  ``execute`` is available only after a completed
strictly loaded ``preflight`` publication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.experiment.paired_tiny_target_artifacts import (  # noqa: E402
    compare_tiny_target_publications,
    execute_tiny_target_audit,
    load_tiny_target_audit_config,
    load_tiny_target_preflight,
    publish_tiny_target_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight",
        help="validate the freeze and publish the catalog without solving",
    )
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    execute = commands.add_parser(
        "execute",
        help="strict-load a preflight and solve every canonical case",
    )
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)

    compare = commands.add_parser(
        "compare",
        help="require two completed publications to be byte-identical",
    )
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "preflight":
        config = load_tiny_target_audit_config(args.config)
        published = publish_tiny_target_preflight(config, args.output)
        return {
            "command": "preflight",
            "status": "PREFLIGHT_COMPLETE",
            "config_fingerprint": published.config_fingerprint,
            "catalog_fingerprint": published.catalog.catalog_fingerprint,
            "equivalence_class_count": len(published.catalog.cases),
            "concrete_placement_count": (
                published.catalog.concrete_placement_count
            ),
            "complete_fingerprint": published.complete_fingerprint,
            "solver_execution_performed": False,
            "training_authorized": False,
            "full_cure_authorized": False,
            "cross_backbone_authorized": False,
        }
    if args.command == "execute":
        config = load_tiny_target_audit_config(args.config)
        preflight = load_tiny_target_preflight(
            args.preflight,
            config=config,
        )
        published = execute_tiny_target_audit(
            config,
            preflight,
            args.output,
        )
        return {
            "command": "execute",
            "status": published.status,
            "config_fingerprint": published.config_fingerprint,
            "catalog_fingerprint": published.catalog.catalog_fingerprint,
            "certificate_set_fingerprint": (
                published.certificate_set_fingerprint
            ),
            "decision_fingerprint": published.decision_fingerprint,
            "complete_fingerprint": published.complete_fingerprint,
            "training_authorized": False,
            "full_cure_authorized": False,
            "cross_backbone_authorized": False,
        }
    if args.command == "compare":
        comparison = compare_tiny_target_publications(
            args.first,
            args.second,
        )
        return {
            "command": "compare",
            **comparison,
        }
    raise RuntimeError("unsupported tiny-target command")


def main(argv: Sequence[str] | None = None) -> None:
    payload = run(parse_args(argv))
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if payload.get("status") == "COMPUTATIONALLY_INCONCLUSIVE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
