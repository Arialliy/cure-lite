"""Exact factual and synthetic supervision masks for CURE-Lite v0.1."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import MatchConfig
from .instances import instances_from_binary_mask, union_instance_masks
from .matching import match_components
from .sampling import choose_uniform_factual_gt_id
from .types import BranchSupervision, InstanceMap, LegalDeletion, MatchResult


def _occupancy_2d(occupancy: Tensor, shape: tuple[int, int]) -> Tensor:
    if not isinstance(occupancy, Tensor):
        raise TypeError("occupancy must be a torch.Tensor")
    if occupancy.dtype != torch.bool:
        raise TypeError("occupancy must be bool")
    result = occupancy.detach().to(device="cpu", dtype=torch.bool)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 2 or tuple(result.shape) != shape:
        raise ValueError("occupancy and GT must have the same [H,W] shape")
    return result.contiguous()


def _gt_union(gt: InstanceMap) -> Tensor:
    return union_instance_masks(gt, gt.ids)


def _validate_factual_state(
    occupancy: Tensor,
    gt: InstanceMap,
    match: MatchResult,
    match_config: MatchConfig,
) -> Tensor:
    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be an InstanceMap")
    if not isinstance(match, MatchResult):
        raise TypeError("match must be a MatchResult")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be MatchConfig")
    occupied = _occupancy_2d(occupancy, gt.shape)
    pred = instances_from_binary_mask(occupied, connectivity=8, min_area=1)
    expected = match_components(pred, gt, match_config)
    if match != expected:
        raise ValueError(
            "match is stale or inconsistent with occupancy, gt, or match_config"
        )
    return occupied


def _full_gt_recoverable_validated(
    occupancy: Tensor,
    gt: InstanceMap,
    gt_id: int,
    before: MatchResult,
    match_config: MatchConfig,
) -> bool:
    oracle_mask = occupancy | gt.by_id(gt_id).mask
    oracle_pred = instances_from_binary_mask(
        oracle_mask,
        connectivity=8,
        min_area=1,
    )
    oracle_match = match_components(oracle_pred, gt, match_config)
    return (
        gt_id in oracle_match.matched_gt_ids
        and before.matched_gt_ids <= oracle_match.matched_gt_ids
    )


def _factual_reachability_catalog_validated(
    occupancy: Tensor,
    gt: InstanceMap,
    match: MatchResult,
    match_config: MatchConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Catalog factual reachability after the caller validated the base state."""

    reachable_ids: list[int] = []
    unreachable_ids: list[int] = []
    for gt_id in sorted(match.unmatched_gt_ids):
        if _full_gt_recoverable_validated(
            occupancy,
            gt,
            gt_id,
            match,
            match_config,
        ):
            reachable_ids.append(gt_id)
        else:
            unreachable_ids.append(gt_id)
    return tuple(reachable_ids), tuple(unreachable_ids)


def full_gt_recoverable(
    occupancy: Tensor,
    gt: InstanceMap,
    gt_id: int,
    before: MatchResult,
    match_config: MatchConfig = MatchConfig(),
) -> bool:
    """Return whether one factual miss is recoverable by adding its full GT.

    This is a conservative full-pipeline diagnostic, not a theoretical oracle
    over arbitrary partial repair masks.  It succeeds only if adding the
    complete GT mask matches that target while retaining every target covered
    by the factual base state.
    """

    occupied = _validate_factual_state(
        occupancy,
        gt,
        before,
        match_config,
    )
    if isinstance(gt_id, bool) or not isinstance(gt_id, int) or gt_id < 1:
        raise ValueError("gt_id must be a positive integer")
    gt.by_id(gt_id)
    if gt_id not in before.unmatched_gt_ids:
        raise ValueError("gt_id must identify a factual unmatched GT")
    return _full_gt_recoverable_validated(
        occupied,
        gt,
        gt_id,
        before,
        match_config,
    )


def factual_oracle_reachable(
    occupancy: Tensor,
    gt: InstanceMap,
    gt_id: int,
    before: MatchResult,
    match_config: MatchConfig = MatchConfig(),
) -> bool:
    """Backward-compatible alias for :func:`full_gt_recoverable`."""

    return full_gt_recoverable(occupancy, gt, gt_id, before, match_config)


def build_factual_supervision(
    occupancy: Tensor,
    gt: InstanceMap,
    match: MatchResult,
    match_config: MatchConfig = MatchConfig(),
    *,
    selected_gt_id: int | None = None,
) -> BranchSupervision:
    """Build one factual target with full-pipeline reachability filtering.

    Reachability is tested independently for every factual miss.  A training
    state nevertheless contains exactly one of those misses: combining several
    individually reachable masks can change connected components and matching,
    so their union is not necessarily jointly reachable.  By default the
    canonical smallest reachable GT ID is selected.  Callers that implement a
    reproducible sampling schedule may explicitly pass another reachable ID.

    ``reachable_gt_ids`` retains the complete diagnostic catalog, while
    ``positive_gt_ids`` contains only the target actually supervised here.
    """

    occupied = _validate_factual_state(occupancy, gt, match, match_config)
    reachable_ids, _ = _factual_reachability_catalog_validated(
        occupied,
        gt,
        match,
        match_config,
    )

    selected = (
        reachable_ids[0]
        if reachable_ids and selected_gt_id is None
        else selected_gt_id
    )
    return build_factual_supervision_from_catalog(
        occupied,
        gt,
        real_miss_ids=tuple(sorted(match.unmatched_gt_ids)),
        reachable_gt_ids=reachable_ids,
        selected_gt_id=selected,
    )


def _canonical_gt_ids(
    values: tuple[int, ...],
    *,
    name: str,
    allowed: frozenset[int],
) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ValueError(f"{name} must contain positive integer IDs")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be sorted and unique")
    if not set(values) <= allowed:
        raise ValueError(f"{name} contains an ID absent from GT")
    return values


def build_factual_supervision_from_catalog(
    occupancy: Tensor,
    gt: InstanceMap,
    *,
    real_miss_ids: tuple[int, ...],
    reachable_gt_ids: tuple[int, ...],
    selected_gt_id: int | None,
) -> BranchSupervision:
    """Materialize one atomic factual state from a verified source catalog.

    Cache consumers must first verify the cached catalogs against the normative
    matcher and full-GT recoverability oracle. This materializer intentionally
    avoids rerunning that expensive oracle for every epoch.
    """

    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be an InstanceMap")
    occupied = _occupancy_2d(occupancy, gt.shape)
    gt_ids = frozenset(gt.ids)
    real = _canonical_gt_ids(
        real_miss_ids,
        name="real_miss_ids",
        allowed=gt_ids,
    )
    reachable = _canonical_gt_ids(
        reachable_gt_ids,
        name="reachable_gt_ids",
        allowed=gt_ids,
    )
    if not set(reachable) <= set(real):
        raise ValueError("reachable_gt_ids must be a subset of real_miss_ids")
    if selected_gt_id is not None and (
        isinstance(selected_gt_id, bool) or not isinstance(selected_gt_id, int)
    ):
        raise TypeError("selected_gt_id must be an integer or None")

    gt_union = _gt_union(gt)
    background = ~gt_union
    writable = ~occupied
    unreachable = tuple(sorted(set(real) - set(reachable)))
    selected_ids: tuple[int, ...] = ()
    if reachable:
        if selected_gt_id not in reachable:
            raise ValueError(
                "selected_gt_id must identify an individually reachable factual miss"
            )
        selected_ids = (selected_gt_id,)
        target = gt.by_id(selected_gt_id).mask & writable
        if not torch.any(target):
            raise ValueError("selected factual target has no writable pixels")
        valid = writable & (background | target)
        branch = "factual_miss"
    elif real:
        if selected_gt_id is not None:
            raise ValueError(
                "selected_gt_id was provided but this state has no reachable miss"
            )
        target = torch.zeros(gt.shape, dtype=torch.bool)
        valid = torch.zeros(gt.shape, dtype=torch.bool)
        branch = "factual_unreachable"
    else:
        if selected_gt_id is not None:
            raise ValueError("selected_gt_id was provided for a no-miss state")
        target = torch.zeros(gt.shape, dtype=torch.bool)
        valid = writable & background
        branch = "factual_no_miss"

    return BranchSupervision(
        occupancy=occupied.unsqueeze(0),
        target=target.to(torch.float32).unsqueeze(0),
        valid_mask=valid.unsqueeze(0),
        branch=branch,
        positive_gt_ids=selected_ids,
        unreachable_gt_ids=unreachable,
        reachable_gt_ids=reachable,
    )


def build_epoch_factual_supervision_from_catalog(
    occupancy: Tensor,
    gt: InstanceMap,
    *,
    real_miss_ids: tuple[int, ...],
    reachable_gt_ids: tuple[int, ...],
    sample_id: str,
    epoch: int,
    global_seed: int,
) -> BranchSupervision:
    """Materialize the epoch's atomic factual state from a full cache catalog.

    Selection is intentionally centralized here so every training variant can
    share the same ``(sample_id, epoch, global_seed)`` schedule.  The returned
    state has at most one positive GT, but retains the complete sorted
    ``reachable_gt_ids`` catalog as metadata.  Empty reachable catalogs still
    materialize the normative ``factual_unreachable`` or ``factual_no_miss``
    branch according to ``real_miss_ids``.
    """

    selected_gt_id = choose_uniform_factual_gt_id(
        reachable_gt_ids,
        sample_id=sample_id,
        epoch=epoch,
        global_seed=global_seed,
    )
    return build_factual_supervision_from_catalog(
        occupancy,
        gt,
        real_miss_ids=real_miss_ids,
        reachable_gt_ids=reachable_gt_ids,
        selected_gt_id=selected_gt_id,
    )


def build_synthetic_supervision(
    deletion: LegalDeletion,
    gt: InstanceMap,
) -> BranchSupervision:
    """Build a synthetic state whose sole positive target is the deleted GT."""

    if not isinstance(deletion, LegalDeletion):
        raise TypeError("deletion must be a LegalDeletion")
    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be an InstanceMap")
    selected = gt.by_id(deletion.gt_id).mask
    occupied = _occupancy_2d(deletion.occupancy_after, gt.shape)
    if deletion.pred_after.shape != gt.shape:
        raise ValueError("deletion and GT shapes differ")
    if not torch.equal(occupied, deletion.pred_after.occupancy):
        raise ValueError("deletion occupancy is inconsistent with pred_after")

    return build_synthetic_supervision_from_catalog(
        occupied,
        gt,
        gt_id=deletion.gt_id,
    )


def build_synthetic_supervision_from_catalog(
    occupancy_after: Tensor,
    gt: InstanceMap,
    *,
    gt_id: int,
) -> BranchSupervision:
    """Materialize a synthetic state from one verified compact candidate.

    Catalog preparation is responsible for proving deletion legality and
    consistency with its post-deletion prediction.  Epoch materialization only
    needs the selected GT identity and resulting occupancy mask.
    """

    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be an InstanceMap")
    if isinstance(gt_id, bool) or not isinstance(gt_id, int) or gt_id < 1:
        raise ValueError("gt_id must be a positive integer")
    selected = gt.by_id(gt_id).mask
    occupied = _occupancy_2d(occupancy_after, gt.shape)

    background = ~_gt_union(gt)
    writable = ~occupied
    target = selected & writable
    valid = writable & (background | selected)
    return BranchSupervision(
        occupancy=occupied.unsqueeze(0),
        target=target.to(torch.float32).unsqueeze(0),
        valid_mask=valid.unsqueeze(0),
        branch="synthetic",
        positive_gt_ids=(gt_id,),
        unreachable_gt_ids=(),
    )


__all__ = [
    "build_epoch_factual_supervision_from_catalog",
    "build_factual_supervision",
    "build_factual_supervision_from_catalog",
    "build_synthetic_supervision",
    "build_synthetic_supervision_from_catalog",
    "factual_oracle_reachable",
    "full_gt_recoverable",
]
