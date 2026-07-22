"""Strict, fixed-horizon training orchestration for cached CURE-Lite states.

This layer deliberately performs no validation-set access, model selection,
checkpointing, or filesystem I/O.  It reconstructs every semantic object from
the cached probability/state tensors before creating epoch-specific training
examples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor, nn

from ..cache.state_cache import StateCacheRecord
from ..config import (
    InterventionConfig,
    MatchConfig,
    OccupancyConfig,
    TrainingConfig,
)
from ..instances import instances_from_binary_mask
from ..intervention import enumerate_legal_deletions
from ..decoder import project_occupancy_to_feature_grid
from ..matching import match_components
from ..occupancy import build_occupancy
from ..sampling import choose_uniform_legal_deletion
from ..supervision import (
    build_epoch_factual_supervision_from_catalog,
    build_factual_supervision,
    build_synthetic_supervision,
)
from ..types import BranchSupervision, InstanceMap, LegalDeletion, MatchResult
from ..train.engine import CURELiteTrainEngine
from ..train.pools import (
    BranchPools,
    StateExample,
    iter_factual_exposure_matched_batches,
    iter_fixed_branch_batches,
)
from ..train.step import BRANCHES


TRAINING_VARIANTS = (
    "factual_only",
    "factual_exposure_matched",
    "uniform_legal",
)

_REQUIRED_BRANCHES = {
    "factual_only": ("factual_miss", "factual_no_miss"),
    "factual_exposure_matched": ("factual_miss", "factual_no_miss"),
    "uniform_legal": ("factual_miss", "factual_no_miss", "synthetic"),
}


def _id_tuple(values: Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in values.tolist())


def _pair_tuple(values: Tensor) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(int(value) for value in row) for row in values.tolist())


def decoder_visible_legal_deletions(
    occupancy: Tensor,
    legal: tuple[LegalDeletion, ...],
    *,
    feature_size: tuple[int, int],
) -> tuple[LegalDeletion, ...]:
    """Keep legal deletions that change the decoder's actual state input."""

    if not isinstance(legal, tuple) or any(
        not isinstance(item, LegalDeletion) for item in legal
    ):
        raise TypeError("legal must be a tuple of LegalDeletion values")
    occupied = torch.as_tensor(occupancy, device="cpu")
    if occupied.ndim == 2:
        occupied = occupied.unsqueeze(0).unsqueeze(0)
    elif occupied.ndim == 3 and occupied.shape[0] == 1:
        occupied = occupied.unsqueeze(0)
    if occupied.ndim != 4 or occupied.shape[:2] != (1, 1):
        raise ValueError("occupancy must contain one binary state")
    if occupied.dtype != torch.bool:
        raise TypeError("occupancy must be bool")
    before = project_occupancy_to_feature_grid(occupied, feature_size)
    visible: list[LegalDeletion] = []
    for deletion in legal:
        after = project_occupancy_to_feature_grid(
            deletion.occupancy_after.unsqueeze(0).unsqueeze(0),
            feature_size,
        )
        if not torch.equal(before, after):
            visible.append(deletion)
    return tuple(visible)


@dataclass(frozen=True)
class TrainingSupportSummary:
    """Independent-image and target support available in one real D_R bundle."""

    source_images: int
    factual_miss_images: int
    factual_no_miss_images: int
    factual_unreachable_images: int
    real_miss_targets: int
    reachable_miss_targets: int
    legal_candidates: int
    decoder_visible_legal_candidates: int
    synthetic_images: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.source_images < 1:
            raise ValueError("source_images must be positive")
        if self.source_images != (
            self.factual_miss_images
            + self.factual_no_miss_images
            + self.factual_unreachable_images
        ):
            raise ValueError("factual image categories must partition D_R")
        if self.reachable_miss_targets > self.real_miss_targets:
            raise ValueError("reachable misses cannot exceed real misses")
        if self.decoder_visible_legal_candidates > self.legal_candidates:
            raise ValueError("visible legal candidates cannot exceed legal candidates")
        if self.synthetic_images > self.source_images:
            raise ValueError("synthetic image support cannot exceed D_R size")

    @property
    def visible_legal_fraction(self) -> float:
        return (
            self.decoder_visible_legal_candidates / self.legal_candidates
            if self.legal_candidates
            else 0.0
        )

    def canonical_payload(self) -> dict[str, int | float]:
        return {
            "source_images": self.source_images,
            "factual_miss_images": self.factual_miss_images,
            "factual_no_miss_images": self.factual_no_miss_images,
            "factual_unreachable_images": self.factual_unreachable_images,
            "real_miss_targets": self.real_miss_targets,
            "reachable_miss_targets": self.reachable_miss_targets,
            "legal_candidates": self.legal_candidates,
            "decoder_visible_legal_candidates": (
                self.decoder_visible_legal_candidates
            ),
            "synthetic_images": self.synthetic_images,
            "visible_legal_fraction": self.visible_legal_fraction,
        }


@dataclass(frozen=True)
class TrainingSupportRequirements:
    """Predeclared minimum support needed for an interpretable Stage-A pilot."""

    minimum_factual_miss_images: int = 1
    minimum_factual_no_miss_images: int = 1
    minimum_synthetic_images: int = 1
    minimum_reachable_miss_targets: int = 1
    minimum_visible_legal_candidates: int = 1
    minimum_visible_legal_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "minimum_factual_miss_images",
            "minimum_factual_no_miss_images",
            "minimum_synthetic_images",
            "minimum_reachable_miss_targets",
            "minimum_visible_legal_candidates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        fraction = self.minimum_visible_legal_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not isfinite(float(fraction))
            or not 0.0 <= float(fraction) <= 1.0
        ):
            raise ValueError("minimum_visible_legal_fraction must lie in [0,1]")
        object.__setattr__(
            self,
            "minimum_visible_legal_fraction",
            float(fraction),
        )

    def canonical_payload(self) -> dict[str, int | float]:
        return {
            "minimum_factual_miss_images": self.minimum_factual_miss_images,
            "minimum_factual_no_miss_images": self.minimum_factual_no_miss_images,
            "minimum_synthetic_images": self.minimum_synthetic_images,
            "minimum_reachable_miss_targets": self.minimum_reachable_miss_targets,
            "minimum_visible_legal_candidates": self.minimum_visible_legal_candidates,
            "minimum_visible_legal_fraction": self.minimum_visible_legal_fraction,
        }

    def require(self, summary: TrainingSupportSummary) -> None:
        if not isinstance(summary, TrainingSupportSummary):
            raise TypeError("summary must be TrainingSupportSummary")
        checks = {
            "factual_miss_images": (
                summary.factual_miss_images,
                self.minimum_factual_miss_images,
            ),
            "factual_no_miss_images": (
                summary.factual_no_miss_images,
                self.minimum_factual_no_miss_images,
            ),
            "synthetic_images": (
                summary.synthetic_images,
                self.minimum_synthetic_images,
            ),
            "reachable_miss_targets": (
                summary.reachable_miss_targets,
                self.minimum_reachable_miss_targets,
            ),
            "decoder_visible_legal_candidates": (
                summary.decoder_visible_legal_candidates,
                self.minimum_visible_legal_candidates,
            ),
            "visible_legal_fraction": (
                summary.visible_legal_fraction,
                self.minimum_visible_legal_fraction,
            ),
        }
        failed = {
            name: {"actual": actual, "minimum": minimum}
            for name, (actual, minimum) in checks.items()
            if actual < minimum
        }
        if failed:
            details = ", ".join(
                f"{name}={values['actual']}<{values['minimum']}"
                for name, values in failed.items()
            )
            raise RuntimeError(f"real D_R support is below Stage-A requirements: {details}")


def summarize_training_support(
    sources: Sequence[CachedTrainingSource],
    *,
    occupancy_config: OccupancyConfig = OccupancyConfig(),
    match_config: MatchConfig = MatchConfig(),
    intervention_config: InterventionConfig = InterventionConfig(),
) -> TrainingSupportSummary:
    """Revalidate and count real D_R support before any decoder update."""

    source_tuple = tuple(sources)
    if not source_tuple:
        raise ValueError("sources cannot be empty")
    if any(not isinstance(source, CachedTrainingSource) for source in source_tuple):
        raise TypeError("sources must contain only CachedTrainingSource values")
    sample_ids = tuple(source.sample_id for source in source_tuple)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("cached training sample IDs must be unique")

    counts = {
        "factual_miss_images": 0,
        "factual_no_miss_images": 0,
        "factual_unreachable_images": 0,
        "real_miss_targets": 0,
        "reachable_miss_targets": 0,
        "legal_candidates": 0,
        "decoder_visible_legal_candidates": 0,
        "synthetic_images": 0,
    }
    for source in sorted(source_tuple, key=lambda item: item.sample_id):
        rebuilt = _recompute_and_validate_state(
            source,
            occupancy_config=occupancy_config,
            match_config=match_config,
            intervention_config=intervention_config,
            rebuild_legal=True,
        )
        real = int(source.state.real_miss_ids.numel())
        reachable = int(source.state.reachable_miss_ids.numel())
        if reachable:
            counts["factual_miss_images"] += 1
        elif real:
            counts["factual_unreachable_images"] += 1
        else:
            counts["factual_no_miss_images"] += 1
        counts["real_miss_targets"] += real
        counts["reachable_miss_targets"] += reachable
        counts["legal_candidates"] += len(rebuilt.legal)
        visible = decoder_visible_legal_deletions(
            source.state.occupancy,
            rebuilt.legal,
            feature_size=tuple(source.feature.shape[-2:]),
        )
        counts["decoder_visible_legal_candidates"] += len(visible)
        if visible:
            counts["synthetic_images"] += 1

    return TrainingSupportSummary(source_images=len(source_tuple), **counts)


def _match_identity(match: MatchResult) -> tuple[tuple[int, int], ...]:
    return tuple((pair.gt_id, pair.pred_id) for pair in match.pairs)


@dataclass(frozen=True, eq=False)
class CachedTrainingSource:
    """One detached base output bound to one normalized state-cache record."""

    sample_id: str
    feature: Tensor
    probability: Tensor
    state: StateCacheRecord

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.state, StateCacheRecord):
            raise TypeError("state must be a StateCacheRecord")
        state = self.state.normalized()
        if state.sample_id != self.sample_id:
            raise ValueError("source and state-cache sample IDs differ")
        if not isinstance(self.feature, Tensor) or not isinstance(
            self.probability, Tensor
        ):
            raise TypeError("feature and probability must be tensors")
        if self.feature.ndim != 4 or self.feature.shape[0] != 1:
            raise ValueError("feature must have shape [1,C,h,w]")
        if self.probability.ndim != 4 or self.probability.shape[:2] != (1, 1):
            raise ValueError("probability must have shape [1,1,H,W]")
        if self.feature.shape[1] < 1 or min(self.feature.shape[-2:]) < 1:
            raise ValueError("feature dimensions must be non-empty")
        if tuple(self.probability.shape[-2:]) != tuple(state.occupancy.shape):
            raise ValueError("probability and state-cache evaluation grids differ")
        if self.feature.dtype != torch.float32:
            raise TypeError("feature must be float32")
        if self.probability.dtype != torch.float32:
            raise TypeError("probability must be float32")
        if self.feature.device.type != "cpu" or self.probability.device.type != "cpu":
            raise ValueError("cached feature and probability tensors must be on CPU")
        if self.feature.requires_grad or self.probability.requires_grad:
            raise ValueError("cached base tensors must be detached")
        if not torch.isfinite(self.feature).all() or not torch.isfinite(
            self.probability
        ).all():
            raise ValueError("cached base tensors must be finite")
        if torch.any((self.probability < 0.0) | (self.probability > 1.0)):
            raise ValueError("probability must lie in [0,1]")
        object.__setattr__(self, "feature", self.feature.contiguous())
        object.__setattr__(self, "probability", self.probability.contiguous())
        object.__setattr__(self, "state", state)


@dataclass(frozen=True)
class _RecomputedState:
    pred: InstanceMap
    gt: InstanceMap
    match: MatchResult
    legal: tuple[LegalDeletion, ...]


def _recompute_and_validate_state(
    source: CachedTrainingSource,
    *,
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    intervention_config: InterventionConfig,
    rebuild_legal: bool,
) -> _RecomputedState:
    state = source.state
    raw_occupancy, _ = build_occupancy(source.probability, occupancy_config)
    # Cache production removes padded/invalid positives *before* CC labeling;
    # mirror that order so invalid bridges cannot merge valid components.
    occupancy = (raw_occupancy & state.image_valid_mask).contiguous()
    pred = instances_from_binary_mask(
        occupancy,
        connectivity=occupancy_config.connectivity,
        min_area=occupancy_config.min_component_area,
    )
    if not torch.equal(occupancy, state.occupancy):
        raise RuntimeError(
            f"cached occupancy disagrees with probability for {source.sample_id!r}"
        )
    if not torch.equal(pred.labels, state.pred_labels):
        raise RuntimeError(
            f"cached prediction labels are not canonical for {source.sample_id!r}"
        )

    gt = instances_from_binary_mask(
        state.gt_labels > 0,
        connectivity=occupancy_config.connectivity,
        min_area=occupancy_config.min_component_area,
    )
    if not torch.equal(gt.labels, state.gt_labels):
        raise RuntimeError(
            f"cached GT labels are not canonical for {source.sample_id!r}"
        )
    match = match_components(pred, gt, match_config)
    if _match_identity(match) != _pair_tuple(state.base_match_pairs):
        raise RuntimeError(
            f"cached base matches disagree with recomputation for {source.sample_id!r}"
        )
    real_miss_ids = tuple(sorted(match.unmatched_gt_ids))
    if real_miss_ids != _id_tuple(state.real_miss_ids):
        raise RuntimeError(
            f"cached real misses disagree with recomputation for {source.sample_id!r}"
        )

    factual_oracle = build_factual_supervision(
        occupancy,
        gt,
        match,
        match_config,
    )
    if factual_oracle.reachable_gt_ids != _id_tuple(state.reachable_miss_ids):
        raise RuntimeError(
            "cached reachable factual catalog disagrees with recomputation for "
            f"{source.sample_id!r}"
        )

    legal: tuple[LegalDeletion, ...] = ()
    if rebuild_legal:
        # U reconstructs actual LegalDeletion objects instead of trusting
        # cached pair IDs. Exact equality rejects fabricated or omitted entries.
        legal = enumerate_legal_deletions(
            pred,
            gt,
            match,
            occupancy,
            match_config=match_config,
            intervention_config=intervention_config,
        )
        legal_identity = tuple((item.gt_id, item.pred_id) for item in legal)
        if legal_identity != _pair_tuple(state.legal_pairs):
            raise RuntimeError(
                "cached legal catalog disagrees with recomputation for "
                f"{source.sample_id!r}"
            )
    return _RecomputedState(pred=pred, gt=gt, match=match, legal=legal)


def _apply_image_valid_mask(
    supervision: BranchSupervision,
    image_valid_mask: Tensor,
) -> BranchSupervision:
    """Intersect spatial supervision while retaining every metadata catalog."""

    valid_2d = torch.as_tensor(image_valid_mask, dtype=torch.bool, device="cpu")
    if valid_2d.ndim == 3 and valid_2d.shape[0] == 1:
        valid_2d = valid_2d[0]
    if valid_2d.ndim != 2 or tuple(valid_2d.shape) != tuple(
        supervision.occupancy.shape[-2:]
    ):
        raise ValueError("image_valid_mask and supervision grids differ")
    valid = supervision.valid_mask & valid_2d.unsqueeze(0)
    target = supervision.target * valid.to(supervision.target.dtype)
    return BranchSupervision(
        occupancy=supervision.occupancy,
        target=target,
        valid_mask=valid,
        branch=supervision.branch,
        positive_gt_ids=supervision.positive_gt_ids,
        unreachable_gt_ids=supervision.unreachable_gt_ids,
        reachable_gt_ids=supervision.reachable_gt_ids,
    )


def build_epoch_branch_pools(
    sources: Sequence[CachedTrainingSource],
    *,
    variant: str,
    epoch: int,
    global_seed: int,
    occupancy_config: OccupancyConfig = OccupancyConfig(),
    match_config: MatchConfig = MatchConfig(),
    intervention_config: InterventionConfig = InterventionConfig(),
) -> BranchPools:
    """Revalidate sources and construct the epoch's F+/F0/(optional) S pools."""

    if variant not in TRAINING_VARIANTS:
        raise ValueError(f"variant must be one of {TRAINING_VARIANTS}")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    if not isinstance(occupancy_config, OccupancyConfig):
        raise TypeError("occupancy_config must be OccupancyConfig")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be MatchConfig")
    if not isinstance(intervention_config, InterventionConfig):
        raise TypeError("intervention_config must be InterventionConfig")
    source_tuple = tuple(sources)
    if not source_tuple:
        raise ValueError("sources cannot be empty")
    if any(not isinstance(source, CachedTrainingSource) for source in source_tuple):
        raise TypeError("sources must contain only CachedTrainingSource values")
    sample_ids = tuple(source.sample_id for source in source_tuple)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("cached training sample IDs must be unique")
    # Pool membership and deterministic draws must not depend on a caller's
    # incidental filesystem/index traversal order.
    source_tuple = tuple(sorted(source_tuple, key=lambda item: item.sample_id))

    factual_miss: list[StateExample] = []
    factual_no_miss: list[StateExample] = []
    synthetic: list[StateExample] = []
    for source in source_tuple:
        rebuilt = _recompute_and_validate_state(
            source,
            occupancy_config=occupancy_config,
            match_config=match_config,
            intervention_config=intervention_config,
            rebuild_legal=variant == "uniform_legal",
        )
        factual = build_epoch_factual_supervision_from_catalog(
            source.state.occupancy,
            rebuilt.gt,
            real_miss_ids=_id_tuple(source.state.real_miss_ids),
            reachable_gt_ids=_id_tuple(source.state.reachable_miss_ids),
            sample_id=source.sample_id,
            epoch=epoch,
            global_seed=global_seed,
        )
        factual = _apply_image_valid_mask(
            factual,
            source.state.image_valid_mask,
        )
        if factual.branch == "factual_miss":
            factual_miss.append(
                StateExample(source.sample_id, source.feature, factual)
            )
        elif factual.branch == "factual_no_miss":
            if not torch.any(factual.valid_mask):
                raise RuntimeError(
                    f"no valid factual negatives for {source.sample_id!r}"
                )
            factual_no_miss.append(
                StateExample(source.sample_id, source.feature, factual)
            )
        elif factual.branch != "factual_unreachable":
            raise AssertionError(f"unexpected factual branch {factual.branch!r}")

        if variant == "uniform_legal":
            visible_legal = decoder_visible_legal_deletions(
                source.state.occupancy,
                rebuilt.legal,
                feature_size=tuple(source.feature.shape[-2:]),
            )
            deletion = choose_uniform_legal_deletion(
                visible_legal,
                sample_id=source.sample_id,
                epoch=epoch,
                global_seed=global_seed,
            )
            if deletion is not None:
                synthetic_state = _apply_image_valid_mask(
                    build_synthetic_supervision(deletion, rebuilt.gt),
                    source.state.image_valid_mask,
                )
                synthetic.append(
                    StateExample(
                        source.sample_id,
                        source.feature,
                        synthetic_state,
                    )
                )

    return BranchPools(
        factual_miss=tuple(factual_miss),
        factual_no_miss=tuple(factual_no_miss),
        synthetic=tuple(synthetic),
    )


def require_training_branch_support(
    pools: BranchPools,
    *,
    variant: str,
) -> None:
    """Reject a formal training variant whose identifying branches are absent.

    Fixed-count sampling intentionally skips empty pools.  That behavior is
    useful for low-level callers, but a formal F/Fx/U comparison would become
    scientifically meaningless if a required branch were silently skipped.
    In particular, U is evidence for the legal-intervention mechanism only
    when factual-positive, factual-negative, and legal-synthetic states all
    exist in the real ``D_R`` bundle.
    """

    if not isinstance(pools, BranchPools):
        raise TypeError("pools must be BranchPools")
    if variant not in TRAINING_VARIANTS:
        raise ValueError(f"variant must be one of {TRAINING_VARIANTS}")
    counts = {branch: len(pools.get(branch)) for branch in BRANCHES}
    missing = tuple(
        branch for branch in _REQUIRED_BRANCHES[variant] if counts[branch] == 0
    )
    if missing:
        formatted = ", ".join(f"{name}={counts[name]}" for name in BRANCHES)
        raise RuntimeError(
            f"{variant} lacks required real D_R branch support: "
            f"{', '.join(missing)} ({formatted})"
        )


@dataclass(frozen=True)
class FixedEpochTrainingLog:
    epoch: int
    pool_sizes: tuple[tuple[str, int], ...]
    metrics: tuple[tuple[str, float | int], ...]

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        expected_pool_order = (
            "factual_miss",
            "factual_no_miss",
            "synthetic",
        )
        if tuple(name for name, _ in self.pool_sizes) != expected_pool_order:
            raise ValueError("pool_sizes must use the canonical F+/F0/S order")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for _, value in self.pool_sizes
        ):
            raise ValueError("pool sizes must be non-negative integers")
        metric_names = tuple(name for name, _ in self.metrics)
        if metric_names != tuple(sorted(set(metric_names))):
            raise ValueError("metric names must be unique and sorted")
        for name, value in self.metrics:
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("training metrics must be numeric")
            if not isfinite(float(value)):
                raise ValueError("training metrics must be finite")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "pool_sizes": dict(self.pool_sizes),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class FixedTrainingLog:
    variant: str
    epochs: int
    steps_per_epoch: int
    epoch_logs: tuple[FixedEpochTrainingLog, ...]

    def __post_init__(self) -> None:
        if self.variant not in TRAINING_VARIANTS:
            raise ValueError(f"variant must be one of {TRAINING_VARIANTS}")
        for name, value in (
            ("epochs", self.epochs),
            ("steps_per_epoch", self.steps_per_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.epoch_logs) != self.epochs:
            raise ValueError("epoch log count must equal epochs")
        for expected_epoch, epoch_log in enumerate(self.epoch_logs):
            if not isinstance(epoch_log, FixedEpochTrainingLog):
                raise TypeError("epoch_logs must contain FixedEpochTrainingLog values")
            if epoch_log.epoch != expected_epoch:
                raise ValueError("epoch logs must be complete and zero-based")
            metrics = dict(epoch_log.metrics)
            if metrics.get("steps") != self.steps_per_epoch:
                raise ValueError("epoch log steps differ from the fixed horizon")
            for branch in BRANCHES:
                for quantity in ("active", "states"):
                    mean_key = f"{branch}/{quantity}"
                    if not (
                        metrics.get(f"{mean_key}_min")
                        == metrics.get(mean_key)
                        == metrics.get(f"{mean_key}_max")
                    ):
                        raise ValueError(
                            f"{branch} {quantity} must be constant on every step"
                        )
            if self.variant in {"factual_only", "factual_exposure_matched"} and dict(
                epoch_log.pool_sizes
            )["synthetic"] != 0:
                raise ValueError(
                    f"{self.variant} logs cannot contain a deletion-synthetic pool"
                )
            if self.variant == "factual_exposure_matched" and (
                metrics.get("synthetic/active") != 1.0
                or not isinstance(metrics.get("synthetic/states"), (int, float))
                or float(metrics["synthetic/states"]) < 1.0
            ):
                raise ValueError("F× must use a non-empty third loss slot")

    def canonical_epoch_logs(self) -> tuple[dict[str, object], ...]:
        """Return the strict JSON-ready records accepted by artifact storage."""

        return tuple(epoch_log.canonical_payload() for epoch_log in self.epoch_logs)


def run_fixed_training(
    decoder: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    sources: Sequence[CachedTrainingSource],
    *,
    variant: str,
    epochs: int,
    steps_per_epoch: int,
    branch_batch_sizes: Mapping[str, int],
    global_seed: int,
    device: torch.device | str = "cpu",
    occupancy_config: OccupancyConfig = OccupancyConfig(),
    match_config: MatchConfig = MatchConfig(),
    intervention_config: InterventionConfig = InterventionConfig(),
    training_config: TrainingConfig = TrainingConfig(),
) -> FixedTrainingLog:
    """Run a fixed number of updates and return immutable training-only logs."""

    for name, value in (("epochs", epochs), ("steps_per_epoch", steps_per_epoch)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be TrainingConfig")
    source_tuple = tuple(sources)
    batch_sizes = dict(branch_batch_sizes)
    engine = CURELiteTrainEngine(decoder, criterion, optimizer, training_config)
    logs: list[FixedEpochTrainingLog] = []
    for epoch in range(epochs):
        pools = build_epoch_branch_pools(
            source_tuple,
            variant=variant,
            epoch=epoch,
            global_seed=global_seed,
            occupancy_config=occupancy_config,
            match_config=match_config,
            intervention_config=intervention_config,
        )
        require_training_branch_support(pools, variant=variant)
        if variant == "factual_exposure_matched":
            batches = iter_factual_exposure_matched_batches(
                pools,
                batch_sizes,
                replacement_count=batch_sizes.get("synthetic", 0),
                epoch=epoch,
                global_seed=global_seed,
                device=device,
                steps=steps_per_epoch,
            )
        else:
            batches = iter_fixed_branch_batches(
                pools,
                batch_sizes,
                epoch=epoch,
                global_seed=global_seed,
                device=device,
                steps=steps_per_epoch,
            )
        summary = engine.run_epoch(batches)
        logs.append(
            FixedEpochTrainingLog(
                epoch=epoch,
                pool_sizes=(
                    ("factual_miss", len(pools.factual_miss)),
                    ("factual_no_miss", len(pools.factual_no_miss)),
                    ("synthetic", len(pools.synthetic)),
                ),
                metrics=tuple(sorted(summary.items())),
            )
        )
    return FixedTrainingLog(
        variant=variant,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        epoch_logs=tuple(logs),
    )


__all__ = [
    "TRAINING_VARIANTS",
    "CachedTrainingSource",
    "FixedEpochTrainingLog",
    "FixedTrainingLog",
    "TrainingSupportRequirements",
    "TrainingSupportSummary",
    "build_epoch_branch_pools",
    "decoder_visible_legal_deletions",
    "require_training_branch_support",
    "run_fixed_training",
    "summarize_training_support",
]
