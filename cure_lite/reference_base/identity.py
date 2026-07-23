"""Verified detector-neutral identity for a project reference-Base run."""

from __future__ import annotations

from pathlib import Path

from ..base_identity import (
    VerifiedBaseRunIdentity,
    _bind_verified_base_run_identity,
)
from ..cache.schema import file_sha256
from ..frozen_base import frozen_base_state_fingerprint
from ..stage_a import BaseRunIdentity
from .training import (
    REFERENCE_BASE_RUN_SCHEMA,
    _strict_json,
    load_reference_base_run,
)
from .adapter import ReferenceBaseAdapter


def load_verified_reference_base_run_identity(
    run_dir: str | Path,
) -> VerifiedBaseRunIdentity:
    """Replay a reference-Base run and expose its neutral registry identity."""

    loaded = load_reference_base_run(run_dir, device="cpu")
    root = loaded.root
    complete_path = root / "COMPLETE.json"
    selection_path = root / "selection.json"
    checkpoint_path = root / "model.safetensors"
    complete = _strict_json(complete_path, name="reference-base COMPLETE")
    selection = _strict_json(selection_path, name="reference-base selection")
    if complete.get("schema_version") != REFERENCE_BASE_RUN_SCHEMA:
        raise RuntimeError("reference-Base completion schema differs")
    adapter = ReferenceBaseAdapter(
        loaded.model,
        loaded.config.preprocess,
        loaded.base_fingerprint,
    )
    identity = BaseRunIdentity(
        producer_schema=REFERENCE_BASE_RUN_SCHEMA,
        base_fingerprint=loaded.base_fingerprint,
        base_state_fingerprint=frozen_base_state_fingerprint(adapter),
        training_run_fingerprint=complete["run_fingerprint"],
        completion_receipt_sha256=file_sha256(complete_path),
        checkpoint_sha256=loaded.checkpoint_sha256,
        selection_fingerprint=selection["selection_fingerprint"],
        source_fingerprint=complete["reference_source_sha256"],
    )
    files = (
        (complete_path, identity.completion_receipt_sha256),
        (selection_path, file_sha256(selection_path)),
        (checkpoint_path, identity.checkpoint_sha256),
    )
    def verify_source() -> None:
        for path, expected_sha256 in files:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"reference-Base identity source is unavailable: {path.name}"
                )
            if file_sha256(path) != expected_sha256:
                raise RuntimeError(
                    f"reference-Base identity source changed: {path.name}"
                )

    return _bind_verified_base_run_identity(identity, verify_source)


__all__ = [
    "load_verified_reference_base_run_identity",
]
