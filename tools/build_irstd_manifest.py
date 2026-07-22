#!/usr/bin/env python3
"""Canonical CLI name for the three-benchmark CURE-Lite manifest builder."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_irstd1k_manifest import (  # noqa: E402
    AUDIT_SCHEMA,
    BuildArtifacts,
    build_irstd_manifest,
    main,
    write_artifacts,
)


__all__ = [
    "AUDIT_SCHEMA",
    "BuildArtifacts",
    "build_irstd_manifest",
    "write_artifacts",
]


if __name__ == "__main__":
    main()
