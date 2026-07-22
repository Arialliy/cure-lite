"""Deterministic uniform sampling of legal single-target deletions."""

from __future__ import annotations

import hashlib

from .types import LegalDeletion


def _validate_key(*, sample_id: str, epoch: int, global_seed: int) -> None:
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")


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
    """Choose from a seed-specific without-replacement catalog cycle.

    Every candidate appears exactly once in each contiguous catalog-length
    cycle.  The seed changes the deterministic permutation, while ``epoch``
    advances through it.  This avoids silently repeating an easy candidate
    while another legal intervention is never exposed during a fixed run.
    """

    if not isinstance(legal_candidates, tuple):
        raise TypeError("legal_candidates must be a tuple")
    if any(not isinstance(item, LegalDeletion) for item in legal_candidates):
        raise TypeError("legal_candidates contains a non-LegalDeletion item")
    _validate_key(sample_id=sample_id, epoch=epoch, global_seed=global_seed)
    if not legal_candidates:
        return None
    identities = tuple((item.gt_id, item.pred_id) for item in legal_candidates)
    if len(identities) != len(set(identities)):
        raise ValueError("legal_candidates must have unique (gt_id, pred_id) identities")
    schedule = tuple(
        sorted(
            legal_candidates,
            key=lambda item: (
                stable_hash(
                    "legal-deletion-cycle-v2",
                    sample_id,
                    global_seed,
                    item.gt_id,
                    item.pred_id,
                ),
                item.gt_id,
                item.pred_id,
            ),
        )
    )
    return schedule[epoch % len(schedule)]


def choose_uniform_factual_gt_id(
    reachable_gt_ids: tuple[int, ...],
    *,
    sample_id: str,
    epoch: int,
    global_seed: int,
) -> int | None:
    """Select one atomic factual target with deterministic cyclic coverage.

    The key deliberately excludes the training variant and model scores, so
    Factual-only, exposure-matched factual, and Uniform-Legal runs share the
    same factual-target schedule when their split, epoch, and seed agree.  A
    sample/seed-specific hash permutation fixes the order, while ``epoch``
    advances through that order.  Consequently, every target is selected
    exactly once in every contiguous ``len(reachable_gt_ids)`` epochs; unlike
    independent per-epoch hashing, the selector cannot repeatedly starve one
    member of a finite catalog.
    """

    if not isinstance(reachable_gt_ids, tuple):
        raise TypeError("reachable_gt_ids must be a tuple")
    if any(
        isinstance(gt_id, bool) or not isinstance(gt_id, int) or gt_id < 1
        for gt_id in reachable_gt_ids
    ):
        raise ValueError("reachable_gt_ids must contain positive integer IDs")
    if reachable_gt_ids != tuple(sorted(set(reachable_gt_ids))):
        raise ValueError("reachable_gt_ids must be sorted and unique")
    _validate_key(sample_id=sample_id, epoch=epoch, global_seed=global_seed)
    if not reachable_gt_ids:
        return None
    schedule = tuple(
        sorted(
            reachable_gt_ids,
            key=lambda gt_id: (
                stable_hash(
                    "factual-target-cycle-v2",
                    sample_id,
                    global_seed,
                    gt_id,
                ),
                gt_id,
            ),
        )
    )
    return schedule[epoch % len(schedule)]


__all__ = [
    "choose_uniform_factual_gt_id",
    "choose_uniform_legal_deletion",
    "stable_hash",
]
