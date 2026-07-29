#!/usr/bin/env python3
"""Run or verify the generated-only v24 GCR-PACRE evidence gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import canonical_json  # noqa: E402
from cure_lite_v24.dataset_free import (  # noqa: E402
    run_gcr_pacre_dataset_free_audit,
    verify_gcr_pacre_dataset_free_receipt,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "gcr_pacre_v24"
    / "dataset_free_receipt.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "new receipt path for generated-only mode "
            f"(default: {_DEFAULT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda:0"),
        default="cpu",
        help="device used for the same-condition efficiency measurements",
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        metavar="RECEIPT",
        help="verify an existing receipt without rerunning any computation",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"receipt contains duplicate key {key!r}")
        result[key] = value
    return result


def _canonical_regular_file(path: Path, *, name: str) -> Path:
    candidate = path.expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if (
        resolved != absolute
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise ValueError(f"{name} must be a canonical regular file")
    return resolved


def _strict_json(path: Path) -> dict[str, object]:
    canonical = _canonical_regular_file(path, name="receipt")
    with canonical.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(
                    f"receipt contains non-finite JSON constant {item}"
                )
            ),
        )
    if not isinstance(value, Mapping):
        raise ValueError("receipt must contain a JSON object")
    return dict(value)


def _prepare_new_output(path: Path) -> Path:
    candidate = path.expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(
            f"dataset-free receipt already exists: {absolute}"
        )
    absolute.parent.mkdir(parents=True, exist_ok=True)
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError(
                "dataset-free receipt path may not traverse a symbolic link"
            )
    resolved_parent = absolute.parent.resolve(strict=True)
    if resolved_parent != absolute.parent:
        raise ValueError(
            "dataset-free receipt parent must be a canonical directory"
        )
    return absolute


def _atomic_write_new_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    output = _prepare_new_output(path)
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    temporary = output.parent / (
        f".{output.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses to replace an existing
        # sealed receipt.  The temporary file is in the same directory.
        os.link(temporary, output, follow_symlinks=False)
        temporary.unlink()
        directory_descriptor = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only is not None:
        receipt = _strict_json(args.verify_only)
        fingerprint = verify_gcr_pacre_dataset_free_receipt(receipt)
        print(
            canonical_json(
                {
                    "mode": "verify-only",
                    "gate_passed": True,
                    "receipt_fingerprint": fingerprint,
                    "efficiency_device": receipt["efficiency_device"],
                }
            )
        )
        return 0

    receipt = run_gcr_pacre_dataset_free_audit(device=args.device)
    _atomic_write_new_json(args.output, receipt)
    print(
        canonical_json(
            {
                "mode": "generated-only",
                "gate_passed": receipt["decision"]["gate_passed"],
                "receipt_fingerprint": receipt["receipt_fingerprint"],
                "efficiency_device": receipt["efficiency_device"],
                "output": str(Path(os.path.abspath(args.output))),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
