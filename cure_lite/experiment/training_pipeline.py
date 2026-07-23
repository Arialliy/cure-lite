"""Strict, fixed-horizon training orchestration for cached CURE-Lite states.

This layer deliberately performs no validation-set access, model selection,
checkpointing, or filesystem I/O.  It validates invariant semantic objects
once, then materializes only epoch-specific choices from a prepared catalog.
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
from ..intervention import _enumerate_legal_deletions_validated
from ..decoder import project_occupancy_to_feature_grid
from ..matching import match_components
from ..occupancy import threshold_occupancy
from ..sampling import (
    choose_uniform_factual_gt_id,
    choose_uniform_legal_identity,
)
from ..supervision import (
    _factual_reachability_catalog_validated,
    build_factual_supervision_from_catalog,
    build_synthetic_supervision_from_catalog,
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
    prepared_catalog: PreparedTrainingCatalog | None = None,
) -> TrainingSupportSummary:
    """Return support generated by the same one-time semantic preparation."""

    catalog = _resolve_prepared_training_catalog(
        sources,
        occupancy_config=occupancy_config,
        match_config=match_config,
        intervention_config=intervention_config,
        prepared_catalog=prepared_catalog,
    )
    return catalog.support_summary


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


def _sorted_training_sources(
    sources: Sequence[CachedTrainingSource],
) -> tuple[CachedTrainingSource, ...]:
    source_tuple = tuple(sources)
    if not source_tuple:
        raise ValueError("sources cannot be empty")
    if any(not isinstance(source, CachedTrainingSource) for source in source_tuple):
        raise TypeError("sources must contain only CachedTrainingSource values")
    sample_ids = tuple(source.sample_id for source in source_tuple)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("cached training sample IDs must be unique")
    return tuple(sorted(source_tuple, key=lambda item: item.sample_id))


def _validate_semantic_configs(
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    intervention_config: InterventionConfig,
) -> None:
    if not isinstance(occupancy_config, OccupancyConfig):
        raise TypeError("occupancy_config must be OccupancyConfig")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be MatchConfig")
    if not isinstance(intervention_config, InterventionConfig):
        raise TypeError("intervention_config must be InterventionConfig")


@dataclass(frozen=True, eq=False)
class PreparedLegalCandidate:
    """Compact epoch-invariant part of one decoder-visible legal deletion."""

    gt_id: int
    pred_id: int
    occupancy_after: Tensor

    def __post_init__(self) -> None:
        for name, value in (("gt_id", self.gt_id), ("pred_id", self.pred_id)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.occupancy_after, Tensor)
            or self.occupancy_after.ndim != 2
        ):
            raise ValueError("occupancy_after must be a [H,W] tensor")
        if (
            self.occupancy_after.device.type != "cpu"
            or self.occupancy_after.dtype != torch.bool
        ):
            raise TypeError("occupancy_after must be a CPU bool tensor")
        object.__setattr__(
            self,
            "occupancy_after",
            self.occupancy_after.detach().clone().contiguous(),
        )

    @property
    def identity(self) -> tuple[int, int]:
        return self.gt_id, self.pred_id


@dataclass(frozen=True, eq=False)
class PreparedTrainingSource:
    """One verified source plus its finite set of reusable training states."""

    source: CachedTrainingSource
    gt: InstanceMap
    real_miss_ids: tuple[int, ...]
    reachable_gt_ids: tuple[int, ...]
    legal_candidates: int
    decoder_visible_legal_candidates: tuple[PreparedLegalCandidate, ...]
    factual_examples: tuple[StateExample, ...]
    factual_no_miss_example: StateExample | None
    synthetic_examples: tuple[StateExample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, CachedTrainingSource):
            raise TypeError("source must be CachedTrainingSource")
        if not isinstance(self.gt, InstanceMap):
            raise TypeError("gt must be InstanceMap")
        if not torch.equal(self.gt.labels, self.source.state.gt_labels):
            raise ValueError("prepared GT differs from its cached source")
        for name, values in (
            ("real_miss_ids", self.real_miss_ids),
            ("reachable_gt_ids", self.reachable_gt_ids),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            if values != tuple(sorted(set(values))) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in values
            ):
                raise ValueError(f"{name} must contain sorted unique positive IDs")
        if not set(self.reachable_gt_ids) <= set(self.real_miss_ids):
            raise ValueError("reachable_gt_ids must be a subset of real_miss_ids")
        if self.real_miss_ids != _id_tuple(self.source.state.real_miss_ids):
            raise ValueError("prepared real misses differ from their cached source")
        if self.reachable_gt_ids != _id_tuple(
            self.source.state.reachable_miss_ids
        ):
            raise ValueError("prepared reachable misses differ from their cached source")
        if (
            isinstance(self.legal_candidates, bool)
            or not isinstance(self.legal_candidates, int)
            or self.legal_candidates < 0
        ):
            raise ValueError("legal_candidates must be a non-negative integer")
        visible = self.decoder_visible_legal_candidates
        if not isinstance(visible, tuple) or any(
            not isinstance(item, PreparedLegalCandidate) for item in visible
        ):
            raise TypeError(
                "decoder_visible_legal_candidates must contain prepared candidates"
            )
        identities = tuple(item.identity for item in visible)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("prepared visible legal identities must be sorted and unique")
        cached_legal_identities = _pair_tuple(self.source.state.legal_pairs)
        if self.legal_candidates != len(cached_legal_identities):
            raise ValueError("prepared legal count differs from its cached source")
        if not set(identities) <= set(cached_legal_identities):
            raise ValueError("prepared visible candidates are absent from cached legal pairs")
        if len(visible) > self.legal_candidates:
            raise ValueError("visible legal candidates cannot exceed all legal candidates")
        if any(item.occupancy_after.shape != self.gt.labels.shape for item in visible):
            raise ValueError("prepared candidate and GT shapes differ")

        factual = self.factual_examples
        if not isinstance(factual, tuple) or any(
            not isinstance(item, StateExample) for item in factual
        ):
            raise TypeError("factual_examples must contain StateExample values")
        if len(factual) != len(self.reachable_gt_ids):
            raise ValueError("factual templates must align with reachable GT IDs")
        unreachable = tuple(
            sorted(set(self.real_miss_ids) - set(self.reachable_gt_ids))
        )
        for gt_id, example in zip(
            self.reachable_gt_ids,
            factual,
            strict=True,
        ):
            supervision = example.supervision
            if (
                example.sample_id != self.source.sample_id
                or example.feature is not self.source.feature
                or supervision.branch != "factual_miss"
                or supervision.positive_gt_ids != (gt_id,)
                or supervision.reachable_gt_ids != self.reachable_gt_ids
                or supervision.unreachable_gt_ids != unreachable
            ):
                raise ValueError("prepared factual template metadata differs")

        no_miss = self.factual_no_miss_example
        if not self.real_miss_ids:
            if not isinstance(no_miss, StateExample):
                raise ValueError("a no-miss source requires one factual template")
            if (
                no_miss.sample_id != self.source.sample_id
                or no_miss.feature is not self.source.feature
                or no_miss.supervision.branch != "factual_no_miss"
            ):
                raise ValueError("prepared no-miss template metadata differs")
        elif no_miss is not None:
            raise ValueError("a source with real misses cannot carry a no-miss template")

        synthetic = self.synthetic_examples
        if not isinstance(synthetic, tuple) or any(
            not isinstance(item, StateExample) for item in synthetic
        ):
            raise TypeError("synthetic_examples must contain StateExample values")
        if len(synthetic) != len(visible):
            raise ValueError("synthetic templates must align with visible candidates")
        for candidate, example in zip(visible, synthetic, strict=True):
            supervision = example.supervision
            if (
                example.sample_id != self.source.sample_id
                or example.feature is not self.source.feature
                or supervision.branch != "synthetic"
                or supervision.positive_gt_ids != (candidate.gt_id,)
            ):
                raise ValueError("prepared synthetic template metadata differs")

    @property
    def sample_id(self) -> str:
        return self.source.sample_id


@dataclass(frozen=True, eq=False)
class PreparedTrainingCatalog:
    """Immutable, process-local semantics for one exact cached source bundle."""

    sources: tuple[CachedTrainingSource, ...]
    entries: tuple[PreparedTrainingSource, ...]
    occupancy_config: OccupancyConfig
    match_config: MatchConfig
    intervention_config: InterventionConfig
    support_summary: TrainingSupportSummary

    def __post_init__(self) -> None:
        _validate_semantic_configs(
            self.occupancy_config,
            self.match_config,
            self.intervention_config,
        )
        canonical_sources = _sorted_training_sources(self.sources)
        if self.sources != canonical_sources:
            raise ValueError("prepared catalog sources must be sorted by sample_id")
        if not isinstance(self.entries, tuple) or len(self.entries) != len(self.sources):
            raise ValueError("prepared entries must align one-to-one with sources")
        if any(
            not isinstance(entry, PreparedTrainingSource)
            for entry in self.entries
        ):
            raise TypeError("entries must contain only PreparedTrainingSource values")
        if any(
            entry.source is not source
            for source, entry in zip(self.sources, self.entries, strict=True)
        ):
            raise ValueError("prepared entries are bound to different source objects")
        if not isinstance(self.support_summary, TrainingSupportSummary):
            raise TypeError("support_summary must be TrainingSupportSummary")
        expected_summary = TrainingSupportSummary(
            source_images=len(self.sources),
            factual_miss_images=sum(bool(entry.reachable_gt_ids) for entry in self.entries),
            factual_no_miss_images=sum(
                not entry.real_miss_ids for entry in self.entries
            ),
            factual_unreachable_images=sum(
                bool(entry.real_miss_ids) and not entry.reachable_gt_ids
                for entry in self.entries
            ),
            real_miss_targets=sum(len(entry.real_miss_ids) for entry in self.entries),
            reachable_miss_targets=sum(
                len(entry.reachable_gt_ids) for entry in self.entries
            ),
            legal_candidates=sum(entry.legal_candidates for entry in self.entries),
            decoder_visible_legal_candidates=sum(
                len(entry.decoder_visible_legal_candidates)
                for entry in self.entries
            ),
            synthetic_images=sum(
                bool(entry.decoder_visible_legal_candidates)
                for entry in self.entries
            ),
        )
        if self.support_summary != expected_summary:
            raise ValueError("support summary differs from prepared catalog entries")

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(source.sample_id for source in self.sources)

    @property
    def summary(self) -> TrainingSupportSummary:
        """Backward-friendly short name for the co-generated support summary."""

        return self.support_summary

    def require_compatible(
        self,
        sources: Sequence[CachedTrainingSource],
        *,
        occupancy_config: OccupancyConfig,
        match_config: MatchConfig,
        intervention_config: InterventionConfig,
    ) -> None:
        """Reject reuse with another source object bundle or semantic config."""

        canonical_sources = _sorted_training_sources(sources)
        _validate_semantic_configs(
            occupancy_config,
            match_config,
            intervention_config,
        )
        if len(canonical_sources) != len(self.sources) or any(
            provided is not prepared
            for provided, prepared in zip(
                canonical_sources,
                self.sources,
                strict=True,
            )
        ):
            raise ValueError("prepared_catalog is bound to different sources")
        if (
            occupancy_config != self.occupancy_config
            or match_config != self.match_config
            or intervention_config != self.intervention_config
        ):
            raise ValueError("prepared_catalog is bound to different semantic configs")


@dataclass(frozen=True)
class _RecomputedState:
    pred: InstanceMap
    gt: InstanceMap
    match: MatchResult
    real_miss_ids: tuple[int, ...]
    reachable_gt_ids: tuple[int, ...]
    legal: tuple[LegalDeletion, ...]


def _recompute_and_validate_state(
    source: CachedTrainingSource,
    *,
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    intervention_config: InterventionConfig,
) -> _RecomputedState:
    state = source.state
    # Threshold only.  The previous build_occupancy call performed a complete
    # connected-component pass before the image-valid mask made that pass
    # unusable; the canonical pass below is the only decomposition required.
    raw_occupancy = threshold_occupancy(
        source.probability,
        occupancy_config.threshold,
    )[0, 0].to(device="cpu", dtype=torch.bool)
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

    reachable_gt_ids, _ = _factual_reachability_catalog_validated(
        occupancy,
        gt,
        match,
        match_config,
    )
    if reachable_gt_ids != _id_tuple(state.reachable_miss_ids):
        raise RuntimeError(
            "cached reachable factual catalog disagrees with recomputation for "
            f"{source.sample_id!r}"
        )

    legal = _enumerate_legal_deletions_validated(
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
    return _RecomputedState(
        pred=pred,
        gt=gt,
        match=match,
        real_miss_ids=real_miss_ids,
        reachable_gt_ids=reachable_gt_ids,
        legal=legal,
    )


def prepare_training_catalog(
    sources: Sequence[CachedTrainingSource],
    *,
    occupancy_config: OccupancyConfig = OccupancyConfig(),
    match_config: MatchConfig = MatchConfig(),
    intervention_config: InterventionConfig = InterventionConfig(),
) -> PreparedTrainingCatalog:
    """Strictly validate invariant semantics once for all variants and epochs."""

    _validate_semantic_configs(
        occupancy_config,
        match_config,
        intervention_config,
    )
    source_tuple = _sorted_training_sources(sources)
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
    entries: list[PreparedTrainingSource] = []
    for source in source_tuple:
        rebuilt = _recompute_and_validate_state(
            source,
            occupancy_config=occupancy_config,
            match_config=match_config,
            intervention_config=intervention_config,
        )
        visible = decoder_visible_legal_deletions(
            source.state.occupancy,
            rebuilt.legal,
            feature_size=tuple(source.feature.shape[-2:]),
        )
        compact_visible = tuple(
            PreparedLegalCandidate(
                gt_id=item.gt_id,
                pred_id=item.pred_id,
                occupancy_after=item.occupancy_after,
            )
            for item in visible
        )
        factual_examples = tuple(
            StateExample(
                source.sample_id,
                source.feature,
                _apply_image_valid_mask(
                    build_factual_supervision_from_catalog(
                        source.state.occupancy,
                        rebuilt.gt,
                        real_miss_ids=rebuilt.real_miss_ids,
                        reachable_gt_ids=rebuilt.reachable_gt_ids,
                        selected_gt_id=gt_id,
                    ),
                    source.state.image_valid_mask,
                ),
            )
            for gt_id in rebuilt.reachable_gt_ids
        )
        factual_no_miss_example: StateExample | None = None
        if not rebuilt.real_miss_ids:
            no_miss_supervision = _apply_image_valid_mask(
                build_factual_supervision_from_catalog(
                    source.state.occupancy,
                    rebuilt.gt,
                    real_miss_ids=(),
                    reachable_gt_ids=(),
                    selected_gt_id=None,
                ),
                source.state.image_valid_mask,
            )
            if not torch.any(no_miss_supervision.valid_mask):
                raise RuntimeError(
                    f"no valid factual negatives for {source.sample_id!r}"
                )
            factual_no_miss_example = StateExample(
                source.sample_id,
                source.feature,
                no_miss_supervision,
            )
        synthetic_examples = tuple(
            StateExample(
                source.sample_id,
                source.feature,
                _apply_image_valid_mask(
                    build_synthetic_supervision_from_catalog(
                        candidate.occupancy_after,
                        rebuilt.gt,
                        gt_id=candidate.gt_id,
                    ),
                    source.state.image_valid_mask,
                ),
            )
            for candidate in compact_visible
        )
        entries.append(
            PreparedTrainingSource(
                source=source,
                gt=rebuilt.gt,
                real_miss_ids=rebuilt.real_miss_ids,
                reachable_gt_ids=rebuilt.reachable_gt_ids,
                legal_candidates=len(rebuilt.legal),
                decoder_visible_legal_candidates=compact_visible,
                factual_examples=factual_examples,
                factual_no_miss_example=factual_no_miss_example,
                synthetic_examples=synthetic_examples,
            )
        )
        real = len(rebuilt.real_miss_ids)
        reachable = len(rebuilt.reachable_gt_ids)
        if reachable:
            counts["factual_miss_images"] += 1
        elif real:
            counts["factual_unreachable_images"] += 1
        else:
            counts["factual_no_miss_images"] += 1
        counts["real_miss_targets"] += real
        counts["reachable_miss_targets"] += reachable
        counts["legal_candidates"] += len(rebuilt.legal)
        counts["decoder_visible_legal_candidates"] += len(compact_visible)
        if compact_visible:
            counts["synthetic_images"] += 1

    support_summary = TrainingSupportSummary(
        source_images=len(source_tuple),
        **counts,
    )
    return PreparedTrainingCatalog(
        sources=source_tuple,
        entries=tuple(entries),
        occupancy_config=occupancy_config,
        match_config=match_config,
        intervention_config=intervention_config,
        support_summary=support_summary,
    )


def _resolve_prepared_training_catalog(
    sources: Sequence[CachedTrainingSource],
    *,
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    intervention_config: InterventionConfig,
    prepared_catalog: PreparedTrainingCatalog | None,
) -> PreparedTrainingCatalog:
    if prepared_catalog is None:
        return prepare_training_catalog(
            sources,
            occupancy_config=occupancy_config,
            match_config=match_config,
            intervention_config=intervention_config,
        )
    if not isinstance(prepared_catalog, PreparedTrainingCatalog):
        raise TypeError("prepared_catalog must be PreparedTrainingCatalog or None")
    prepared_catalog.require_compatible(
        sources,
        occupancy_config=occupancy_config,
        match_config=match_config,
        intervention_config=intervention_config,
    )
    return prepared_catalog


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


def build_epoch_branch_pools_from_catalog(
    catalog: PreparedTrainingCatalog,
    *,
    variant: str,
    epoch: int,
    global_seed: int,
) -> BranchPools:
    """Materialize one epoch without repeating invariant semantic work."""

    if not isinstance(catalog, PreparedTrainingCatalog):
        raise TypeError("catalog must be PreparedTrainingCatalog")
    if variant not in TRAINING_VARIANTS:
        raise ValueError(f"variant must be one of {TRAINING_VARIANTS}")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")

    factual_miss: list[StateExample] = []
    factual_no_miss: list[StateExample] = []
    synthetic: list[StateExample] = []
    for entry in catalog.entries:
        source = entry.source
        selected_gt_id = choose_uniform_factual_gt_id(
            entry.reachable_gt_ids,
            sample_id=source.sample_id,
            epoch=epoch,
            global_seed=global_seed,
        )
        if selected_gt_id is not None:
            factual_miss.append(
                entry.factual_examples[
                    entry.reachable_gt_ids.index(selected_gt_id)
                ]
            )
        elif not entry.real_miss_ids:
            if entry.factual_no_miss_example is None:
                raise AssertionError("prepared no-miss template is absent")
            factual_no_miss.append(entry.factual_no_miss_example)

        if variant == "uniform_legal":
            candidates = entry.decoder_visible_legal_candidates
            selected_identity = choose_uniform_legal_identity(
                tuple(item.identity for item in candidates),
                sample_id=source.sample_id,
                epoch=epoch,
                global_seed=global_seed,
            )
            if selected_identity is not None:
                candidate_index = next(
                    index
                    for index, item in enumerate(candidates)
                    if item.identity == selected_identity
                )
                synthetic.append(entry.synthetic_examples[candidate_index])

    return BranchPools(
        factual_miss=tuple(factual_miss),
        factual_no_miss=tuple(factual_no_miss),
        synthetic=tuple(synthetic),
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
    """Compatibility wrapper: strictly prepare once, then materialize."""

    if variant not in TRAINING_VARIANTS:
        raise ValueError(f"variant must be one of {TRAINING_VARIANTS}")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    catalog = prepare_training_catalog(
        sources,
        occupancy_config=occupancy_config,
        match_config=match_config,
        intervention_config=intervention_config,
    )
    return build_epoch_branch_pools_from_catalog(
        catalog,
        variant=variant,
        epoch=epoch,
        global_seed=global_seed,
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
    prepared_catalog: PreparedTrainingCatalog | None = None,
) -> FixedTrainingLog:
    """Run a fixed number of updates and return immutable training-only logs."""

    for name, value in (("epochs", epochs), ("steps_per_epoch", steps_per_epoch)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if variant not in TRAINING_VARIANTS:
        raise ValueError(f"variant must be one of {TRAINING_VARIANTS}")
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be TrainingConfig")
    source_tuple = tuple(sources)
    catalog = _resolve_prepared_training_catalog(
        source_tuple,
        occupancy_config=occupancy_config,
        match_config=match_config,
        intervention_config=intervention_config,
        prepared_catalog=prepared_catalog,
    )
    batch_sizes = dict(branch_batch_sizes)
    engine = CURELiteTrainEngine(decoder, criterion, optimizer, training_config)
    logs: list[FixedEpochTrainingLog] = []
    for epoch in range(epochs):
        pools = build_epoch_branch_pools_from_catalog(
            catalog,
            variant=variant,
            epoch=epoch,
            global_seed=global_seed,
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
    "PreparedLegalCandidate",
    "PreparedTrainingCatalog",
    "PreparedTrainingSource",
    "TrainingSupportRequirements",
    "TrainingSupportSummary",
    "build_epoch_branch_pools",
    "build_epoch_branch_pools_from_catalog",
    "decoder_visible_legal_deletions",
    "prepare_training_catalog",
    "require_training_branch_support",
    "run_fixed_training",
    "summarize_training_support",
]
