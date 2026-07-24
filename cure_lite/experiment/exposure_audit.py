"""Exact D_R-only synthetic exposure replay for P0-D."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from math import ceil, exp
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..sampling import stable_hash
from ..splits import SplitManifest
from ..train.pools import _draw
from .p0_protocol import P0ExposureConfig, P0OverlapConfig
from .training_pipeline import (
    PreparedTrainingCatalog,
    build_epoch_branch_pools_from_catalog,
)


P0_D_SCHEMA = "cure-lite-p0-d-exposure-v1"
LegalIdentity = tuple[str, int, int]
FactualIdentity = tuple[str, int]


def _legal_catalog(
    catalog: PreparedTrainingCatalog,
) -> tuple[
    tuple[LegalIdentity, ...],
    dict[int, LegalIdentity],
    dict[LegalIdentity, str],
]:
    identities: list[LegalIdentity] = []
    by_example: dict[int, LegalIdentity] = {}
    source_by_identity: dict[LegalIdentity, str] = {}
    for entry in catalog.entries:
        for candidate, example in zip(
            entry.decoder_visible_legal_candidates,
            entry.synthetic_examples,
            strict=True,
        ):
            identity = (entry.sample_id, candidate.gt_id, candidate.pred_id)
            identities.append(identity)
            by_example[id(example)] = identity
            source_by_identity[identity] = entry.sample_id
    ordered = tuple(sorted(identities))
    if len(ordered) != len(set(ordered)) or len(by_example) != len(ordered):
        raise RuntimeError("legal target catalog is not one-to-one")
    return ordered, by_example, source_by_identity


def _sequence_update(
    digest: hashlib._Hash,
    identity: Sequence[object],
) -> None:
    digest.update(
        json.dumps(
            list(identity),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _mapping_sequence_update(
    digest: hashlib._Hash,
    factual: FactualIdentity,
    legal: LegalIdentity,
) -> None:
    digest.update(
        json.dumps(
            {
                "factual": list(factual),
                "legal": list(legal),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _gini(counts: Sequence[int]) -> float:
    ordered = sorted(int(value) for value in counts)
    total = sum(ordered)
    if not total:
        return 0.0
    size = len(ordered)
    numerator = sum(
        (2 * index - size - 1) * value
        for index, value in enumerate(ordered, start=1)
    )
    return numerator / (size * total)


def _count_stats(
    counts: Mapping[object, int],
    *,
    top_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, object]:
    ordered = sorted(
        ((identity, int(value)) for identity, value in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    values = [value for _, value in ordered]
    total = sum(values)
    square_sum = sum(value * value for value in values)
    result: dict[str, object] = {
        "population": len(values),
        "total_exposures": total,
        "unique_exposed": sum(value > 0 for value in values),
        "zero_exposure": sum(value == 0 for value in values),
        "minimum_count": min(values, default=0),
        "maximum_count": max(values, default=0),
        "maximum_share": max(values, default=0) / total if total else 0.0,
        "ess": total * total / square_sum if square_sum else 0.0,
        "normalized_ess": (
            total * total / square_sum / len(values)
            if square_sum and values
            else 0.0
        ),
        "gini": _gini(values),
    }
    for top in top_values:
        selected = ordered[: min(top, len(ordered))]
        result[f"top{top}"] = {
            "share": (
                sum(value for _, value in selected) / total if total else 0.0
            ),
            "items": [
                {
                    "identity": (
                        list(identity)
                        if isinstance(identity, tuple)
                        else identity
                    ),
                    "count": value,
                }
                for identity, value in selected
            ],
        }
    return result


def _historical_sequence(
    catalog: PreparedTrainingCatalog,
    *,
    variant: str,
    seed: int,
    config: P0ExposureConfig,
    legal_identities: Sequence[LegalIdentity],
    identity_by_example: Mapping[int, LegalIdentity],
) -> dict[str, object]:
    target_counts = Counter({identity: 0 for identity in legal_identities})
    sources = sorted({identity[0] for identity in legal_identities})
    source_counts = Counter({source: 0 for source in sources})
    factual_origin_targets = tuple(
        (entry.sample_id, gt_id)
        for entry in catalog.entries
        for gt_id in entry.reachable_gt_ids
    )
    factual_origin_sources = sorted(
        {identity[0] for identity in factual_origin_targets}
    )
    origin_target_counts = Counter(
        {identity: 0 for identity in factual_origin_targets}
    )
    origin_source_counts = Counter(
        {source: 0 for source in factual_origin_sources}
    )
    sequence = hashlib.sha256()
    mapping_sequence = hashlib.sha256()
    epoch_unique_targets: list[int] = []
    epoch_unique_sources: list[int] = []
    expected = config.epochs * config.steps_per_epoch * config.synthetic_batch
    for epoch in range(config.epochs):
        pools = build_epoch_branch_pools_from_catalog(
            catalog,
            variant=variant,
            epoch=epoch,
            global_seed=seed,
        )
        epoch_targets: set[LegalIdentity] = set()
        epoch_sources: set[str] = set()
        epoch_factual_origins: tuple[FactualIdentity, ...] = ()
        if variant == "miss_aligned_legal":
            resolved: list[FactualIdentity] = []
            for example in pools.factual_miss:
                positive = example.supervision.positive_gt_ids
                if len(positive) != 1:
                    raise RuntimeError(
                        "miss-aligned factual origin is not atomic"
                    )
                resolved.append((example.sample_id, positive[0]))
            epoch_factual_origins = tuple(resolved)
            if len(epoch_factual_origins) != len(pools.synthetic):
                raise RuntimeError(
                    "miss-aligned origin and synthetic pools differ"
                )
        for step in range(config.steps_per_epoch):
            selected = _draw(
                pools.synthetic,
                config.synthetic_batch,
                branch="synthetic",
                epoch=epoch,
                step=step,
                global_seed=seed,
            )
            for draw, example in enumerate(selected):
                pool_index = (
                    stable_hash("synthetic", epoch, step, draw, seed)
                    % len(pools.synthetic)
                )
                if example is not pools.synthetic[pool_index]:
                    raise RuntimeError(
                        "exposure replay index differs from training _draw"
                    )
                identity = identity_by_example[id(example)]
                target_counts[identity] += 1
                source_counts[identity[0]] += 1
                epoch_targets.add(identity)
                epoch_sources.add(identity[0])
                _sequence_update(sequence, identity)
                if variant == "miss_aligned_legal":
                    origin = epoch_factual_origins[pool_index]
                    origin_target_counts[origin] += 1
                    origin_source_counts[origin[0]] += 1
                    _mapping_sequence_update(
                        mapping_sequence,
                        origin,
                        identity,
                    )
        epoch_unique_targets.append(len(epoch_targets))
        epoch_unique_sources.append(len(epoch_sources))
    if sum(target_counts.values()) != expected:
        raise RuntimeError("historical exposure replay has the wrong event count")
    result = {
        "variant": variant,
        "seed": seed,
        "epochs": config.epochs,
        "steps_per_epoch": config.steps_per_epoch,
        "synthetic_batch": config.synthetic_batch,
        "events": expected,
        "sequence_fingerprint": sequence.hexdigest(),
        "target": _count_stats(target_counts),
        "source_image": _count_stats(source_counts),
        "per_epoch_unique": {
            "target": {
                "minimum": min(epoch_unique_targets),
                "mean": sum(epoch_unique_targets) / len(epoch_unique_targets),
                "maximum": max(epoch_unique_targets),
            },
            "source_image": {
                "minimum": min(epoch_unique_sources),
                "mean": sum(epoch_unique_sources) / len(epoch_unique_sources),
                "maximum": max(epoch_unique_sources),
            },
        },
        "target_counts": [
            {"identity": list(identity), "count": target_counts[identity]}
            for identity in legal_identities
        ],
        "source_counts": [
            {"sample_id": source, "count": source_counts[source]}
            for source in sources
        ],
    }
    if variant == "miss_aligned_legal":
        result["alignment_catalog_fingerprint"] = (
            catalog.miss_alignment_fingerprint
        )
        result["mapping_sequence_fingerprint"] = mapping_sequence.hexdigest()
        result["factual_origin_target"] = _count_stats(origin_target_counts)
        result["factual_origin_source"] = _count_stats(origin_source_counts)
        result["factual_origin_target_counts"] = [
            {"identity": list(identity), "count": origin_target_counts[identity]}
            for identity in factual_origin_targets
        ]
        result["factual_origin_counts"] = [
            {"sample_id": source, "count": origin_source_counts[source]}
            for source in factual_origin_sources
        ]
    return result


def _robust_scaled(
    factual: Tensor,
    legal: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    median = torch.median(legal, dim=0).values
    deviation = torch.abs(legal - median)
    mad = torch.median(deviation, dim=0).values
    maxdev = torch.max(deviation, dim=0).values
    floor = torch.maximum(torch.ones_like(median), torch.abs(median)) * 1e-12
    use_mad = mad > floor
    use_maxdev = (~use_mad) & (maxdev > floor)
    constant = (~use_mad) & (~use_maxdev)
    scale = torch.where(
        use_mad,
        mad,
        torch.where(use_maxdev, maxdev, floor),
    )
    return (
        (factual - median) / scale,
        (legal - median) / scale,
        median,
        scale,
        constant,
        use_maxdev,
    )


def _kth_group_distance(
    query: Tensor,
    query_group: str,
    legal: Tensor,
    legal_groups: Sequence[str],
    k: int,
) -> float:
    distances = torch.linalg.vector_norm(legal - query, dim=1)
    by_group: dict[str, float] = {}
    for distance, group in zip(distances.tolist(), legal_groups, strict=True):
        if group == query_group:
            continue
        by_group[group] = min(float(distance), by_group.get(group, float("inf")))
    if len(by_group) < k:
        raise RuntimeError("candidate proposal lacks k source-disjoint groups")
    return sorted(by_group.values())[k - 1]


def _ess(probabilities: Tensor) -> float:
    return 1.0 / float(torch.sum(probabilities.square()))


def _proposal_constraints(
    probabilities: Tensor,
    identities: Sequence[LegalIdentity],
    config: P0ExposureConfig,
) -> bool:
    target_count = len(identities)
    if (
        torch.any(probabilities <= 0.0)
        or _ess(probabilities)
        < config.target_ess_minimum_fraction_of_legal * target_count
        or float(probabilities.max())
        > config.target_maximum_uniform_multiple / target_count
    ):
        return False
    sources = sorted({identity[0] for identity in identities})
    source_probability = torch.tensor(
        [
            sum(
                float(probabilities[index])
                for index, identity in enumerate(identities)
                if identity[0] == source
            )
            for source in sources
        ],
        dtype=torch.float64,
    )
    ordered = torch.sort(source_probability, descending=True).values
    if (
        _ess(source_probability)
        < config.source_ess_minimum_fraction_of_legal_sources * len(sources)
        or float(source_probability.max())
        > config.source_maximum_uniform_multiple / len(sources)
        or float(ordered[:5].sum()) > config.source_top5_maximum_share
        or float(ordered[:10].sum()) > config.source_top10_maximum_share
    ):
        return False
    return True


def _integer_mass(
    probabilities: Tensor,
    total: int,
    identities: Sequence[LegalIdentity],
) -> tuple[int, ...]:
    raw = probabilities * total
    base = torch.floor(raw).to(torch.int64)
    remaining = total - int(base.sum())
    fractions = [
        (float(raw[index] - base[index]), identities[index], index)
        for index in range(len(identities))
    ]
    fractions.sort(key=lambda item: (-item[0], item[1]))
    for _, _, index in fractions[:remaining]:
        base[index] += 1
    values = tuple(int(value) for value in base.tolist())
    if sum(values) != total or any(value < 1 for value in values):
        raise RuntimeError("candidate integer mass is invalid")
    return values


def _build_candidate_proposal(
    *,
    factual_values: Tensor,
    legal_values: Tensor,
    factual_sample_ids: Sequence[str],
    factual_groups: Sequence[str],
    legal_identities: Sequence[LegalIdentity],
    legal_groups: Sequence[str],
    overlap: P0OverlapConfig,
    config: P0ExposureConfig,
) -> dict[str, object]:
    (
        factual,
        legal,
        descriptor_median,
        descriptor_scale,
        descriptor_constant,
        descriptor_maxdev_fallback,
    ) = _robust_scaled(factual_values, legal_values)
    if len(factual_sample_ids) != len(factual):
        raise RuntimeError("candidate factual identities and values differ")
    kth = [
        _kth_group_distance(
            factual[index],
            factual_groups[index],
            legal,
            legal_groups,
            overlap.knn_k,
        )
        for index in range(len(factual))
    ]
    bandwidth = sorted(kth)[len(kth) // 2]
    if bandwidth <= 0.0:
        raise RuntimeError("candidate proposal bandwidth is non-positive")
    factual_weight = torch.full(
        (len(factual_sample_ids),),
        1.0 / len(factual_sample_ids),
        dtype=torch.float64,
    )
    affinity = torch.zeros(len(legal), dtype=torch.float64)
    for factual_index in range(len(factual)):
        distances2 = torch.sum(
            (legal - factual[factual_index]).square(),
            dim=1,
        )
        kernel = torch.exp(-distances2 / (2.0 * bandwidth * bandwidth))
        same_group = torch.tensor(
            [
                group == factual_groups[factual_index]
                for group in legal_groups
            ],
            dtype=torch.bool,
        )
        kernel[same_group] = 0.0
        affinity += factual_weight[factual_index] * kernel
    if float(affinity.sum()) <= 0.0:
        raise RuntimeError("candidate proposal has zero total affinity")
    informed = affinity / affinity.sum()
    legal_source_counts = Counter(identity[0] for identity in legal_identities)
    legal_sources = sorted(legal_source_counts)
    uniform = torch.tensor(
        [
            1.0 / len(legal_sources) / legal_source_counts[identity[0]]
            for identity in legal_identities
        ],
        dtype=torch.float64,
    )
    proposal_config = config.candidate_marginal_proposal
    maximum_lambda = 1.0 - proposal_config.minimum_uniform_floor
    denominator = proposal_config.lambda_grid_denominator
    maximum_index = int(maximum_lambda * denominator + 1e-12)
    selected_lambda: float | None = None
    selected: Tensor | None = None
    for index in range(maximum_index, -1, -1):
        value = index / denominator
        candidate = (1.0 - value) * uniform + value * informed
        candidate /= candidate.sum()
        if _proposal_constraints(candidate, legal_identities, config):
            selected_lambda = value
            selected = candidate
            break
    if selected is None or selected_lambda is None:
        raise RuntimeError("candidate proposal has no feasible marginal distribution")
    integer_mass = _integer_mass(
        selected,
        proposal_config.integer_mass_total,
        legal_identities,
    )
    return {
        "factual_weighting": (
            proposal_config.factual_weighting
        ),
        "descriptor_standardization": {
            "rule": overlap.robust_scale_rule,
            "legal_median": [
                float(value) for value in descriptor_median.tolist()
            ],
            "legal_scale": [
                float(value) for value in descriptor_scale.tolist()
            ],
            "maxdev_fallback_dimensions": [
                index
                for index, flag in enumerate(
                    descriptor_maxdev_fallback.tolist()
                )
                if flag
            ],
            "constant_floor_dimensions": [
                index
                for index, flag in enumerate(
                    descriptor_constant.tolist()
                )
                if flag
            ],
        },
        "bandwidth": bandwidth,
        "lambda": selected_lambda,
        "uniform_floor": 1.0 - selected_lambda,
        "probabilities": selected,
        "integer_mass": integer_mass,
        "integer_mass_total": proposal_config.integer_mass_total,
        "proposal_fingerprint": stable_fingerprint(
            {
                "factual_weighting": proposal_config.factual_weighting,
                "descriptor_median": [
                    float(value) for value in descriptor_median.tolist()
                ],
                "descriptor_scale": [
                    float(value) for value in descriptor_scale.tolist()
                ],
                "bandwidth": bandwidth,
                "lambda": selected_lambda,
                "identities": [list(identity) for identity in legal_identities],
                "probabilities": [float(value) for value in selected.tolist()],
                "integer_mass": list(integer_mass),
            }
        ),
    }


def _candidate_sequence(
    proposal: Mapping[str, object],
    identities: Sequence[LegalIdentity],
    *,
    seed: int,
    config: P0ExposureConfig,
) -> dict[str, object]:
    masses = tuple(int(value) for value in proposal["integer_mass"])
    total_mass = int(proposal["integer_mass_total"])
    cumulative: list[int] = []
    running = 0
    for value in masses:
        running += value
        cumulative.append(running)
    if running != total_mass:
        raise RuntimeError("candidate CDF mass differs from its total")
    target_counts = Counter({identity: 0 for identity in identities})
    sources = sorted({identity[0] for identity in identities})
    source_counts = Counter({source: 0 for source in sources})
    sequence = hashlib.sha256()
    epoch_unique_targets: list[int] = []
    epoch_unique_sources: list[int] = []
    for epoch in range(config.epochs):
        epoch_targets: set[LegalIdentity] = set()
        epoch_sources: set[str] = set()
        for step in range(config.steps_per_epoch):
            for draw in range(config.synthetic_batch):
                position = (
                    stable_hash("synthetic", epoch, step, draw, seed)
                    % total_mass
                )
                low, high = 0, len(cumulative)
                while low < high:
                    middle = (low + high) // 2
                    if position < cumulative[middle]:
                        high = middle
                    else:
                        low = middle + 1
                identity = identities[low]
                target_counts[identity] += 1
                source_counts[identity[0]] += 1
                epoch_targets.add(identity)
                epoch_sources.add(identity[0])
                _sequence_update(sequence, identity)
        epoch_unique_targets.append(len(epoch_targets))
        epoch_unique_sources.append(len(epoch_sources))
    target = _count_stats(target_counts)
    source = _count_stats(source_counts)
    target_population = len(identities)
    source_population = len(sources)
    gates = {
        "all_targets_exposed": target["zero_exposure"] == 0,
        "target_ess": (
            float(target["ess"])
            >= config.target_ess_minimum_fraction_of_legal * target_population
        ),
        "target_maximum_share": (
            float(target["maximum_share"])
            <= config.target_maximum_uniform_multiple / target_population
        ),
        "source_ess": (
            float(source["ess"])
            >= config.source_ess_minimum_fraction_of_legal_sources
            * source_population
        ),
        "source_maximum_share": (
            float(source["maximum_share"])
            <= config.source_maximum_uniform_multiple / source_population
        ),
        "source_top5_share": (
            float(source["top5"]["share"])
            <= config.source_top5_maximum_share
        ),
        "source_top10_share": (
            float(source["top10"]["share"])
            <= config.source_top10_maximum_share
        ),
    }
    return {
        "seed": seed,
        "events": config.epochs
        * config.steps_per_epoch
        * config.synthetic_batch,
        "sequence_fingerprint": sequence.hexdigest(),
        "target": target,
        "source_image": source,
        "per_epoch_unique": {
            "target": {
                "minimum": min(epoch_unique_targets),
                "mean": sum(epoch_unique_targets) / len(epoch_unique_targets),
                "maximum": max(epoch_unique_targets),
            },
            "source_image": {
                "minimum": min(epoch_unique_sources),
                "mean": sum(epoch_unique_sources) / len(epoch_unique_sources),
                "maximum": max(epoch_unique_sources),
            },
        },
        "gates": gates,
        "pass": all(gates.values()),
        "target_counts": [
            {"identity": list(identity), "count": target_counts[identity]}
            for identity in identities
        ],
        "source_counts": [
            {"sample_id": source_id, "count": source_counts[source_id]}
            for source_id in sources
        ],
    }


def build_p0_d_exposure(
    catalog: PreparedTrainingCatalog,
    manifest: SplitManifest,
    overlap: P0OverlapConfig,
    config: P0ExposureConfig,
    *,
    upstream_pass: bool,
    factual_hand: Tensor,
    legal_hand: Tensor,
) -> dict[str, object]:
    """Replay historical U/M and conditionally evaluate a candidate S schedule."""

    identities, identity_by_example, _ = _legal_catalog(catalog)
    historical: dict[str, object] = {}
    for seed in config.seeds:
        historical[str(seed)] = {
            "U": _historical_sequence(
                catalog,
                variant="uniform_legal",
                seed=seed,
                config=config,
                legal_identities=identities,
                identity_by_example=identity_by_example,
            ),
            "M": _historical_sequence(
                catalog,
                variant="miss_aligned_legal",
                seed=seed,
                config=config,
                legal_identities=identities,
                identity_by_example=identity_by_example,
            ),
        }
    if not upstream_pass:
        candidate: dict[str, object] = {
            "status": "not_evaluated",
            "reason": "P0-A/B/C did not all pass",
            "training_integration": False,
        }
        status = "not_evaluated"
        p0_d_pass: bool | None = None
        failure_decision = "follow_upstream_failure_decision"
    else:
        group_by_sample = {
            record.sample_id: record.group_id
            for record in manifest.records_for("D_R")
        }
        factual_identities = [
            (entry.sample_id, gt_id)
            for entry in catalog.entries
            for gt_id in entry.reachable_gt_ids
        ]
        legal_groups = [
            group_by_sample[identity[0]]
            for identity in identities
        ]
        proposal = _build_candidate_proposal(
            factual_values=factual_hand,
            legal_values=legal_hand,
            factual_sample_ids=[identity[0] for identity in factual_identities],
            factual_groups=[
                group_by_sample[identity[0]] for identity in factual_identities
            ],
            legal_identities=identities,
            legal_groups=legal_groups,
            overlap=overlap,
            config=config,
        )
        runs = {
            str(seed): _candidate_sequence(
                proposal,
                identities,
                seed=seed,
                config=config,
            )
            for seed in config.seeds
        }
        candidate = {
            "status": "evaluated",
            "training_integration": False,
            "proposal": {
                key: (
                    [float(value) for value in item.tolist()]
                    if isinstance(item, Tensor)
                    else list(item)
                    if isinstance(item, tuple)
                    else item
                )
                for key, item in proposal.items()
            },
            "runs": runs,
        }
        p0_d_pass = all(bool(run["pass"]) for run in runs.values())
        status = "pass" if p0_d_pass else "fail"
        failure_decision = (
            None if p0_d_pass else "revise_marginal_sampling_distribution"
        )
    return {
        "schema_version": P0_D_SCHEMA,
        "split": "D_R",
        "historical_alignment_catalog_fingerprint": (
            catalog.miss_alignment_fingerprint
        ),
        "historical_replay": historical,
        "candidate_s": candidate,
        "status": status,
        "p0_d_pass": p0_d_pass,
        "failure_decision": failure_decision,
    }


__all__ = ["P0_D_SCHEMA", "build_p0_d_exposure"]
