"""Paired seed/scene hierarchical bootstrap used by formal Stage A."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
import random
from typing import Iterable, cast

import numpy as np


@dataclass(frozen=True)
class PairedObservation:
    seed_id: str
    scene_id: str
    baseline: float
    method: float

    def __post_init__(self) -> None:
        if not self.seed_id or not self.scene_id:
            raise ValueError("seed_id and scene_id must be non-empty")
        if not isfinite(self.baseline) or not isfinite(self.method):
            raise ValueError("paired values must be finite")

    @property
    def difference(self) -> float:
        return self.method - self.baseline


@dataclass(frozen=True)
class PairedRatioObservation:
    """Paired sufficient statistics for one pipeline seed and test unit.

    Keeping numerator and denominator separate is essential for micro-averaged
    metrics such as RMR, Pd, and pixel FA.  Averaging already-divided scene
    ratios would give a small scene the same weight as a large one and would
    therefore estimate a different quantity.
    """

    seed_id: str
    scene_id: str
    baseline_numerator: float
    baseline_denominator: float
    method_numerator: float
    method_denominator: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.seed_id, str)
            or not self.seed_id
            or not isinstance(self.scene_id, str)
            or not self.scene_id
        ):
            raise ValueError("seed_id and scene_id must be non-empty")
        for name in (
            "baseline_numerator",
            "baseline_denominator",
            "method_numerator",
            "method_denominator",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise TypeError(f"{name} must be a real number")
            value = float(raw)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.baseline_denominator == 0.0 and self.baseline_numerator != 0.0:
            raise ValueError("a zero baseline denominator requires a zero numerator")
        if self.method_denominator == 0.0 and self.method_numerator != 0.0:
            raise ValueError("a zero method denominator requires a zero numerator")


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    replicates: int
    pipeline_seeds: int
    scenes: int

    def supports(self, minimum_effect: float) -> bool:
        """Return true only when the whole interval exceeds a preregistered effect."""

        return self.lower > minimum_effect


def paired_hierarchical_bootstrap(
    observations: Iterable[PairedObservation],
    *,
    replicates: int = 10_000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> BootstrapInterval:
    """Resample full-pipeline seeds, then scene/sequence units within each seed."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    grouped: dict[str, dict[str, PairedObservation]] = {}
    items = tuple(observations)
    if not items:
        raise ValueError("at least one paired observation is required")
    for item in items:
        by_scene = grouped.setdefault(item.seed_id, {})
        if item.scene_id in by_scene:
            raise ValueError(
                f"duplicate paired unit seed={item.seed_id!r}, scene={item.scene_id!r}"
            )
        by_scene[item.scene_id] = item
    seed_ids = sorted(grouped)
    seed_means = [
        sum(item.difference for item in grouped[seed_id].values())
        / len(grouped[seed_id])
        for seed_id in seed_ids
    ]
    estimate = float(sum(seed_means) / len(seed_means))

    rng = random.Random(random_seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seed_means = []
        for _ in seed_ids:
            sampled_seed = seed_ids[rng.randrange(len(seed_ids))]
            scene_values = tuple(
                grouped[sampled_seed][scene_id]
                for scene_id in sorted(grouped[sampled_seed])
            )
            sampled_differences = [
                scene_values[rng.randrange(len(scene_values))].difference
                for _ in scene_values
            ]
            sampled_seed_means.append(
                sum(sampled_differences) / len(sampled_differences)
            )
        draws[replicate] = sum(sampled_seed_means) / len(sampled_seed_means)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail], method="linear")
    return BootstrapInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        replicates=replicates,
        pipeline_seeds=len(seed_ids),
        scenes=len({item.scene_id for item in items}),
    )


def _formal_groups(
    items: tuple[object, ...],
    record_type: type,
) -> dict[str, dict[str, object]]:
    """Validate the formal seed/unit pairing contract and return grouped rows."""

    if not items:
        raise ValueError("formal bootstrap requires paired observations")

    grouped: dict[str, dict[str, object]] = {}
    for item in items:
        if not isinstance(item, record_type):
            raise TypeError(
                f"observations must contain {record_type.__name__} records"
            )
        by_scene = grouped.setdefault(item.seed_id, {})
        if item.scene_id in by_scene:
            raise ValueError(
                f"duplicate paired unit seed={item.seed_id!r}, "
                f"scene={item.scene_id!r}"
            )
        by_scene[item.scene_id] = item

    if len(grouped) < 5:
        raise ValueError("formal Stage A requires at least 5 paired pipeline seeds")

    ordered_seeds = sorted(grouped)
    reference_seed = ordered_seeds[0]
    reference_units = set(grouped[reference_seed])
    for seed_id in ordered_seeds[1:]:
        units = set(grouped[seed_id])
        if units != reference_units:
            missing = sorted(reference_units - units)
            extra = sorted(units - reference_units)
            raise ValueError(
                "formal paired seeds must contain identical scene/sequence units; "
                f"seed={seed_id!r}, reference_seed={reference_seed!r}, "
                f"missing={missing}, extra={extra}"
            )
    return grouped


def formal_paired_hierarchical_bootstrap(
    observations: Iterable[PairedObservation],
    *,
    replicates: int = 10_000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> BootstrapInterval:
    """Run the Stage-A bootstrap only after enforcing its formal pairing gates.

    Formal CURE-Lite evidence requires at least five independently trained
    full-pipeline seeds.  Every seed must evaluate exactly the same set of
    scene/sequence units; otherwise a nominally paired comparison can silently
    become an unbalanced one.  The low-level bootstrap above intentionally
    remains available for pilot and diagnostic use without these formal gates.
    """

    items = tuple(observations)
    _formal_groups(items, PairedObservation)

    return paired_hierarchical_bootstrap(
        tuple(sorted(items, key=lambda item: (item.seed_id, item.scene_id))),
        replicates=replicates,
        confidence=confidence,
        random_seed=random_seed,
    )


def _micro_ratio_difference(
    observations: Iterable[PairedRatioObservation],
    *,
    require_information: bool,
) -> float:
    items = tuple(observations)
    baseline_numerator = sum(item.baseline_numerator for item in items)
    baseline_denominator = sum(item.baseline_denominator for item in items)
    method_numerator = sum(item.method_numerator for item in items)
    method_denominator = sum(item.method_denominator for item in items)
    if require_information and (
        baseline_denominator == 0.0 or method_denominator == 0.0
    ):
        raise ValueError(
            "formal ratio bootstrap requires positive aggregate method and "
            "baseline denominators"
        )
    baseline_ratio = (
        baseline_numerator / baseline_denominator
        if baseline_denominator
        else 0.0
    )
    method_ratio = (
        method_numerator / method_denominator if method_denominator else 0.0
    )
    return method_ratio - baseline_ratio


def _equal_seed_ratio_difference(
    grouped: dict[str, dict[str, PairedRatioObservation]],
) -> float:
    """Return the mean of per-seed micro-ratio differences.

    A full-pipeline seed is the top-level experimental unit.  Pooling counts
    across seeds would silently give a seed with more anchor misses (or GTs)
    more weight than another seed.  Counts are therefore pooled only within a
    seed; the resulting paired seed effects are averaged with equal weight.
    """

    effects = []
    for seed_id in sorted(grouped):
        effects.append(
            _micro_ratio_difference(
                grouped[seed_id].values(), require_information=True
            )
        )
    return float(sum(effects) / len(effects))


def _resample_informative_units(
    rng: random.Random,
    units: tuple[PairedRatioObservation, ...],
) -> tuple[PairedRatioObservation, ...]:
    """Resample one seed's units while keeping a ratio statistic defined.

    Sparse RMR data can produce an all-zero-denominator scene draw even though
    the original seed contains anchor misses.  Such a draw has no defined RMR;
    assigning it the numeric value zero would bias the hierarchical estimate.
    Conditional redraws preserve the requested within-seed bootstrap while
    avoiding that artificial zero contribution.
    """

    for _ in range(10_000):
        sampled = tuple(units[rng.randrange(len(units))] for _ in units)
        try:
            _micro_ratio_difference(sampled, require_information=True)
        except ValueError:
            continue
        return sampled
    raise RuntimeError(
        "unable to draw an informative within-seed ratio bootstrap sample"
    )


def formal_paired_hierarchical_ratio_bootstrap(
    observations: Iterable[PairedRatioObservation],
    *,
    replicates: int = 10_000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap a paired difference of micro-averaged ratios.

    Full-pipeline seeds are sampled first.  Within every sampled seed, the
    common scene/sequence units are sampled with replacement and their four
    sufficient statistics are pooled before computing that seed's paired
    micro-ratio difference.  Seed effects are then averaged with equal weight.
    """

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    items = tuple(observations)
    grouped_raw = _formal_groups(items, PairedRatioObservation)
    grouped: dict[str, dict[str, PairedRatioObservation]] = {
        seed_id: {
            scene_id: cast(PairedRatioObservation, item)
            for scene_id, item in by_scene.items()
        }
        for seed_id, by_scene in grouped_raw.items()
    }
    # Validate every top-level seed independently and estimate the paired
    # effect with equal seed weight.
    estimate = _equal_seed_ratio_difference(grouped)

    seed_ids = tuple(sorted(grouped))
    scene_values = {
        seed_id: tuple(
            grouped[seed_id][scene_id] for scene_id in sorted(grouped[seed_id])
        )
        for seed_id in seed_ids
    }
    rng = random.Random(random_seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seed_effects: list[float] = []
        for _ in seed_ids:
            sampled_seed = seed_ids[rng.randrange(len(seed_ids))]
            units = scene_values[sampled_seed]
            sampled_units = _resample_informative_units(rng, units)
            sampled_seed_effects.append(
                _micro_ratio_difference(
                    sampled_units, require_information=True
                )
            )
        draws[replicate] = sum(sampled_seed_effects) / len(
            sampled_seed_effects
        )

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail], method="linear")
    return BootstrapInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        replicates=replicates,
        pipeline_seeds=len(seed_ids),
        scenes=len(scene_values[seed_ids[0]]),
    )


__all__ = [
    "BootstrapInterval",
    "PairedObservation",
    "PairedRatioObservation",
    "formal_paired_hierarchical_bootstrap",
    "formal_paired_hierarchical_ratio_bootstrap",
    "paired_hierarchical_bootstrap",
]
