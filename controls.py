"""Stage A supervision/sampling controls; these do not change CURE-Lite U."""

from __future__ import annotations

import torch
from torch import Tensor

from .instances import union_instance_masks
from .sampling import stable_hash
from .types import BranchSupervision, InstanceMap, LegalDeletion


def build_parallel_all_gt_supervision(
    occupancy: Tensor,
    gt: InstanceMap,
) -> BranchSupervision:
    """P control: supervise every GT outside the same fixed occupancy gate."""

    occupied = torch.as_tensor(occupancy, dtype=torch.bool, device="cpu")
    if occupied.ndim == 3 and occupied.shape[0] == 1:
        occupied = occupied[0]
    if occupied.ndim != 2 or tuple(occupied.shape) != gt.shape:
        raise ValueError("occupancy and GT grids must match")
    writable = ~occupied
    target = union_instance_masks(gt, gt.ids) & writable
    positive_gt_ids = tuple(
        gt_id for gt_id in gt.ids if bool(torch.any(gt.by_id(gt_id).mask & writable))
    )
    return BranchSupervision(
        occupancy=occupied.unsqueeze(0),
        target=target.to(torch.float32).unsqueeze(0),
        valid_mask=writable.unsqueeze(0),
        branch="factual_miss" if target.any() else "factual_no_miss",
        positive_gt_ids=positive_gt_ids,
    )


def choose_score_hard_deletion(
    candidates: tuple[LegalDeletion, ...],
    probability: Tensor,
    pred: InstanceMap,
    *,
    reduction: str = "mean",
) -> LegalDeletion | None:
    """S control: pick the legal component with the lowest frozen base score."""

    if not candidates:
        return None
    if reduction not in {"mean", "max"}:
        raise ValueError("score reduction must be 'mean' or 'max'")
    p = torch.as_tensor(probability, dtype=torch.float32, device="cpu")
    if p.ndim == 3 and p.shape[0] == 1:
        p = p[0]
    if p.ndim != 2 or tuple(p.shape) != pred.shape:
        raise ValueError("probability and prediction grids must match")
    if not torch.isfinite(p).all() or torch.any((p < 0) | (p > 1)):
        raise ValueError("probability must be finite and lie in [0,1]")

    scored = []
    for candidate in candidates:
        component = pred.by_id(candidate.pred_id)
        values = p[component.mask]
        score = values.mean() if reduction == "mean" else values.max()
        scored.append((float(score), candidate.gt_id, candidate.pred_id, candidate))
    return min(scored, key=lambda item: item[:3])[3]


def exposure_matched_indices(
    factual_positive_count: int,
    target_positive_forwards: int,
    *,
    epoch: int,
    global_seed: int,
) -> tuple[int, ...]:
    """F× control: deterministic replacement sampling of factual positive states."""

    if factual_positive_count < 1:
        if target_positive_forwards == 0:
            return ()
        raise ValueError("cannot exposure-match without factual positive states")
    if target_positive_forwards < 0:
        raise ValueError("target_positive_forwards must be non-negative")
    return tuple(
        stable_hash("factual-exposure", epoch, draw, global_seed)
        % factual_positive_count
        for draw in range(target_positive_forwards)
    )
