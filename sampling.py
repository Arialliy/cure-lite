"""Deterministic uniform sampling of legal single-target deletions."""

from __future__ import annotations

import hashlib

from .types import LegalDeletion


def stable_hash(*parts: object) -> int:
    """Return a process-independent 64-bit hash of typed, length-prefixed parts."""

    digest = hashlib.sha256()
    for part in parts:
        type_name = type(part).__qualname__.encode("utf-8")
        value = str(part).encode("utf-8")
        digest.update(len(type_name).to_bytes(4, "big"))
        digest.update(type_name)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def choose_uniform_legal_deletion(
    legal_candidates: tuple[LegalDeletion, ...],
    *,
    sample_id: str,
    epoch: int,
    global_seed: int,
) -> LegalDeletion | None:
    """Choose one candidate reproducibly, or ``None`` for an empty set."""

    if not isinstance(legal_candidates, tuple):
        raise TypeError("legal_candidates must be a tuple")
    if any(not isinstance(item, LegalDeletion) for item in legal_candidates):
        raise TypeError("legal_candidates contains a non-LegalDeletion item")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    if not legal_candidates:
        return None
    index = stable_hash(sample_id, epoch, global_seed) % len(legal_candidates)
    return legal_candidates[index]
