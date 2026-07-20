"""Miss-propensity matching for counterfactual coverage interventions.

All predictions are image-disjoint and source-only.  The estimator is a small
L2-regularized logistic regression implemented in float64 so the method does
not acquire a hidden scikit-learn dependency or a trainable inference module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math

import torch
from torch import Tensor

from ..sampling import stable_hash
from ..splits import SplitManifest
from ..types import InstanceMap
from .config import PropensityConfig
from .descriptors import dilate_mask
from .protocol import CUREProtocol, tensor_content_fingerprint
from .types import (
    _INTERVENTION_CATALOG_SEAL,
    _OOF_PROPENSITY_SEAL,
    CUREInterventionCatalog,
    EligibleSampleCatalog,
    OOFPropensityResult,
    PropensityEstimate,
    TargetDescriptor,
    WeightedCounterfactualCandidate,
)


def miss_odds(probability: float, config: PropensityConfig) -> float:
    """Return clipped miss odds used for normalized candidate sampling."""

    if not isinstance(config, PropensityConfig):
        raise TypeError("config must be PropensityConfig")
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly inside (0,1)")
    clipped = min(max(probability, config.clip_epsilon), 1.0 - config.clip_epsilon)
    return min(clipped / (1.0 - clipped), config.max_odds)


def _fit_logistic(
    features: Tensor,
    labels: Tensor,
    config: PropensityConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have shapes [N,D] and [N]")
    if features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("logistic fitting requires at least two rows and one feature")
    unique = torch.unique(labels)
    if not torch.equal(unique, torch.tensor((0.0, 1.0), dtype=torch.float64)):
        raise ValueError("every training fold must contain covered and missed targets")

    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    standardized = (features - mean) / scale
    design = torch.cat(
        (torch.ones((features.shape[0], 1), dtype=torch.float64), standardized),
        dim=1,
    )
    beta = torch.zeros(design.shape[1], dtype=torch.float64)
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * config.l2
    regularizer[0, 0] = 0.0

    converged = False
    for _ in range(config.max_iterations):
        logits = torch.clamp(design @ beta, min=-30.0, max=30.0)
        probability = torch.sigmoid(logits)
        weight = torch.clamp(probability * (1.0 - probability), min=1e-9)
        gradient = design.T @ (probability - labels) + regularizer @ beta
        hessian = design.T @ (weight[:, None] * design) + regularizer
        try:
            step = torch.linalg.solve(hessian, gradient)
        except RuntimeError:
            jitter = torch.eye(hessian.shape[0], dtype=hessian.dtype) * 1e-8
            step = torch.linalg.solve(hessian + jitter, gradient)
        beta_next = beta - step
        if float(torch.max(torch.abs(beta_next - beta))) <= config.tolerance:
            beta = beta_next
            converged = True
            break
        beta = beta_next
    if not converged:
        raise RuntimeError(
            "propensity logistic solver did not converge within max_iterations"
        )
    return beta, mean, scale


def _validate_manifest_groups(
    items: tuple[TargetDescriptor, ...],
    manifest: SplitManifest,
) -> None:
    """Bind descriptor sample/group IDs to the frozen source split manifest."""

    if not isinstance(manifest, SplitManifest):
        raise TypeError("manifest must be SplitManifest")
    source_records = {record.sample_id: record for record in manifest.records_for("D_R")}
    for item in items:
        try:
            record = source_records[item.sample_id]
        except KeyError as error:
            raise ValueError(
                f"descriptor sample {item.sample_id!r} is not assigned to D_R"
            ) from error
        if item.group_id != record.group_id:
            raise ValueError(
                f"descriptor group for {item.sample_id!r} differs from the manifest"
            )

    # Every known provenance key must map to one OOF group.  Otherwise two
    # crops/frames from the same scene could leak across folds despite having
    # different nominal group_id strings.
    owner_by_key: dict[tuple[str, str], str] = {}
    for record in manifest.records_for("D_R"):
        for key in record.grouping_keys():
            owner = owner_by_key.setdefault(key, record.group_id)
            if owner != record.group_id:
                kind, value = key
                raise ValueError(
                    f"D_R {kind}={value!r} spans multiple OOF group_id values"
                )


def _validate_catalogs(
    sample_catalogs: tuple[EligibleSampleCatalog, ...],
    protocol: CUREProtocol,
) -> tuple[TargetDescriptor, ...]:
    if not isinstance(protocol, CUREProtocol):
        raise TypeError("protocol must be CUREProtocol")
    if not isinstance(sample_catalogs, tuple) or not sample_catalogs or any(
        not isinstance(item, EligibleSampleCatalog) for item in sample_catalogs
    ):
        raise TypeError(
            "sample_catalogs must be a non-empty EligibleSampleCatalog tuple"
        )
    sample_ids = tuple(item.sample_id for item in sample_catalogs)
    expected_ids = tuple(
        sorted(
            sample_id
            for sample_id, split, _ in protocol.manifest_membership
            if split == "D_R"
        )
    )
    if sample_ids != expected_ids:
        raise ValueError(
            "canonical propensity input must contain exactly one catalog per D_R sample"
        )
    for catalog in sample_catalogs:
        catalog.validate_receipt()
        protocol.assert_sample(
            catalog.sample_id, split="D_R", group_id=catalog.group_id
        )
        if (
            catalog.protocol_fingerprint != protocol.fingerprint
            or catalog.base_fingerprint != protocol.base_fingerprint
            or catalog.occupancy_threshold
            != protocol.residual_config.occupancy_threshold
            or catalog.suppression_radius
            != protocol.residual_config.suppression_radius
        ):
            raise ValueError("eligible catalog differs from the frozen CURE protocol")
    return tuple(
        sorted(
            (
                descriptor
                for catalog in sample_catalogs
                for descriptor in catalog.descriptors
            ),
            key=lambda item: item.key,
        )
    )


def _descriptor_fingerprint(items: tuple[TargetDescriptor, ...]) -> str:
    payload = repr(
        tuple(
            (
                item.sample_id,
                item.group_id,
                item.gt_id,
                item.role,
                tuple(float(value) for value in item.values),
            )
            for item in items
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _predict_logistic(
    features: Tensor,
    model: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    beta, mean, scale = model
    standardized = (features - mean) / scale
    design = torch.cat(
        (torch.ones((features.shape[0], 1), dtype=torch.float64), standardized),
        dim=1,
    )
    return torch.sigmoid(torch.clamp(design @ beta, min=-30.0, max=30.0))


def _fold_assignment(
    group_ids: Sequence[str],
    config: PropensityConfig,
) -> dict[str, int]:
    unique = set(group_ids)
    if len(unique) != len(group_ids):
        raise ValueError("group_ids passed to fold assignment must be unique")
    if len(unique) < config.folds:
        raise ValueError("number of source images must be at least the fold count")
    ordered = sorted(
        unique,
        key=lambda group_id: (
            stable_hash("cure-propensity-fold", config.seed, group_id),
            group_id,
        ),
    )
    return {group_id: index % config.folds for index, group_id in enumerate(ordered)}


def cross_fit_miss_propensity(
    sample_catalogs: tuple[EligibleSampleCatalog, ...],
    *,
    protocol: CUREProtocol,
    manifest: SplitManifest,
) -> OOFPropensityResult:
    """Estimate ``P(missed | descriptor)`` with group-disjoint cross-fitting.

    Every target receives exactly one prediction from a model that saw no
    target from the same source group.  Failure to obtain both classes in a
    training fold is explicit; silently falling back to in-sample or uniform
    scores would invalidate the uncensoring claim.
    """

    protocol.validate_manifest(manifest)
    config = protocol.propensity_config
    items = _validate_catalogs(sample_catalogs, protocol)
    if not items:
        raise ValueError("canonical eligible catalogs contain no propensity targets")
    _validate_manifest_groups(items, manifest)
    keys = tuple(item.key for item in items)
    if len(set(keys)) != len(keys):
        raise ValueError("target descriptor keys must be unique")
    feature_sizes = {int(item.values.shape[0]) for item in items}
    if len(feature_sizes) != 1:
        raise ValueError("all target descriptors must use the same schema")

    sample_to_group: dict[str, str] = {}
    for item in items:
        previous = sample_to_group.setdefault(item.sample_id, item.group_id)
        if previous != item.group_id:
            raise ValueError("one sample_id may not belong to multiple groups")
    group_ids = tuple(sorted({item.group_id for item in items}))
    assignment = _fold_assignment(group_ids, config)
    raw_by_key: dict[tuple[str, int], tuple[int, float]] = {}
    for fold in range(config.folds):
        train = tuple(item for item in items if assignment[item.group_id] != fold)
        validation = tuple(item for item in items if assignment[item.group_id] == fold)
        if not validation:
            raise RuntimeError(f"propensity fold {fold} has no validation images")
        train_x = torch.stack([item.values for item in train])
        train_y = torch.tensor(
            [float(item.missed) for item in train], dtype=torch.float64
        )
        model = _fit_logistic(train_x, train_y, config)
        validation_x = torch.stack([item.values for item in validation])
        predictions = _predict_logistic(validation_x, model)
        for item, probability in zip(validation, predictions, strict=True):
            raw_by_key[item.key] = (fold, float(probability))

    estimates: list[PropensityEstimate] = []
    probability_clipped_count = 0
    odds_capped_count = 0
    covered_weights: list[float] = []
    raw_predictions: list[float] = []
    binary_labels: list[float] = []
    for item in sorted(items, key=lambda value: value.key):
        fold, raw = raw_by_key[item.key]
        clipped = min(max(raw, config.clip_epsilon), 1.0 - config.clip_epsilon)
        raw_odds = clipped / (1.0 - clipped)
        weight = min(raw_odds, config.max_odds) if item.covered else None
        if raw != clipped:
            probability_clipped_count += 1
        if item.covered and raw_odds > config.max_odds:
            odds_capped_count += 1
        if weight is not None:
            covered_weights.append(weight)
        raw_predictions.append(raw)
        binary_labels.append(float(item.missed))
        estimates.append(
            PropensityEstimate(
                sample_id=item.sample_id,
                group_id=item.group_id,
                gt_id=item.gt_id,
                role=item.role,
                fold=fold,
                raw_probability=raw,
                clipped_probability=clipped,
                sampling_weight=weight,
            )
        )
    if not covered_weights:
        raise ValueError("at least one covered target is required for uncensoring")
    weights = torch.tensor(covered_weights, dtype=torch.float64)
    effective_sample_size = float(weights.sum().square() / weights.square().sum())
    factual_features = torch.stack([item.values for item in items if item.missed])
    legal_features = torch.stack([item.values for item in items if item.covered])
    if not factual_features.shape[0] or not legal_features.shape[0]:
        raise ValueError("eligible propensity universe requires both target roles")
    factual_mean = factual_features.mean(dim=0)
    factual_var = factual_features.var(dim=0, unbiased=False)
    legal_mean = legal_features.mean(dim=0)
    legal_var = legal_features.var(dim=0, unbiased=False)
    normalized_weights = weights / weights.sum()
    weighted_legal_mean = (normalized_weights[:, None] * legal_features).sum(dim=0)
    weighted_legal_var = (
        normalized_weights[:, None]
        * (legal_features - weighted_legal_mean).square()
    ).sum(dim=0)
    denominator_before = torch.sqrt(torch.clamp((factual_var + legal_var) / 2.0, min=1e-12))
    denominator_after = torch.sqrt(
        torch.clamp((factual_var + weighted_legal_var) / 2.0, min=1e-12)
    )
    smd_before = (factual_mean - legal_mean) / denominator_before
    smd_after = (factual_mean - weighted_legal_mean) / denominator_after
    predictions = torch.tensor(raw_predictions, dtype=torch.float64)
    labels = torch.tensor(binary_labels, dtype=torch.float64)
    factual_predictions = predictions[labels == 1.0]
    legal_predictions = predictions[labels == 0.0]
    factual_range = (
        float(factual_predictions.min()),
        float(factual_predictions.max()),
    )
    legal_range = (float(legal_predictions.min()), float(legal_predictions.max()))
    overlap_low = max(factual_range[0], legal_range[0])
    overlap_high = min(factual_range[1], legal_range[1])
    overlap = (overlap_low, overlap_high) if overlap_low <= overlap_high else None
    return OOFPropensityResult(
        estimates=tuple(estimates),
        fold_by_group=tuple(sorted(assignment.items())),
        descriptor_fingerprint=_descriptor_fingerprint(items),
        manifest_fingerprint=manifest.fingerprint,
        protocol_fingerprint=protocol.fingerprint,
        effective_sample_size=effective_sample_size,
        probability_clipped_fraction=probability_clipped_count / len(items),
        odds_capped_fraction=odds_capped_count / len(covered_weights),
        brier_score=float((predictions - labels).square().mean()),
        factual_fraction=float(labels.mean()),
        factual_probability_range=factual_range,
        legal_probability_range=legal_range,
        overlap_interval=overlap,
        max_sampling_weight=float(weights.max()),
        clip_epsilon=config.clip_epsilon,
        max_odds=config.max_odds,
        smd_before=tuple(float(value) for value in smd_before),
        smd_after=tuple(float(value) for value in smd_after),
        _seal=_OOF_PROPENSITY_SEAL,
    )


def bind_weighted_candidates(
    sample_catalogs: tuple[EligibleSampleCatalog, ...],
    gt_by_sample: Mapping[str, InstanceMap],
    propensity: OOFPropensityResult,
    *,
    protocol: CUREProtocol,
) -> CUREInterventionCatalog:
    """Bind canonical eligible catalogs and OOF odds into one receipt."""

    if not isinstance(propensity, OOFPropensityResult):
        raise TypeError("propensity must be OOFPropensityResult")
    propensity.validate_receipt()
    descriptor_items = _validate_catalogs(sample_catalogs, protocol)
    if (
        propensity.protocol_fingerprint != protocol.fingerprint
        or propensity.manifest_fingerprint != protocol.manifest_fingerprint
    ):
        raise ValueError("OOF propensity differs from the frozen CURE protocol")
    descriptor_keys = tuple(item.key for item in descriptor_items)
    estimate_keys = tuple(item.key for item in propensity.estimates)
    if descriptor_keys != estimate_keys or any(
        descriptor.role != estimate.role
        for descriptor, estimate in zip(
            descriptor_items, propensity.estimates, strict=True
        )
    ):
        raise ValueError("eligible descriptor universe and OOF estimates differ")
    if _descriptor_fingerprint(descriptor_items) != propensity.descriptor_fingerprint:
        raise ValueError("eligible descriptor values differ from OOF fitting records")
    estimate_by_key = propensity.by_key()
    legal_estimate_keys = {
        item.key for item in propensity.estimates if item.covered
    }
    candidates: list[WeightedCounterfactualCandidate] = []
    seen_candidate_keys: set[tuple[str, int, int]] = set()
    seen_target_keys: set[tuple[str, int]] = set()
    for sample_catalog in sample_catalogs:
        sample_id = sample_catalog.sample_id
        try:
            gt = gt_by_sample[sample_id]
        except KeyError as error:
            raise ValueError(f"sample {sample_id!r} has no GT instance map") from error
        if not isinstance(gt, InstanceMap):
            raise TypeError("gt_by_sample values must be InstanceMap")
        if tensor_content_fingerprint("gt_labels", gt.labels) != sample_catalog.gt_fingerprint:
            raise ValueError(f"GT for sample {sample_id!r} differs from its catalog")
        deletions = sample_catalog.legal_deletions
        for deletion in deletions:
            candidate_key = (sample_id, deletion.gt_id, deletion.pred_id)
            if candidate_key in seen_candidate_keys:
                raise ValueError(f"duplicate legal candidate key {candidate_key!r}")
            seen_candidate_keys.add(candidate_key)
            target_key = (sample_id, deletion.gt_id)
            if target_key in seen_target_keys:
                raise ValueError(
                    f"multiple legal deletions map to propensity target {target_key!r}"
                )
            seen_target_keys.add(target_key)
            editable = ~dilate_mask(
                deletion.occupancy_after,
                protocol.residual_config.suppression_radius,
            )
            if not bool(torch.any(gt.by_id(deletion.gt_id).mask & editable)):
                raise ValueError(
                    f"legal candidate {candidate_key!r} has no support under the "
                    "full-CURE suppression radius"
                )
            key = target_key
            try:
                estimate = estimate_by_key[key]
            except KeyError as error:
                raise ValueError(f"legal candidate {key!r} has no OOF propensity") from error
            if not estimate.covered or estimate.sampling_weight is None:
                raise ValueError(f"legal candidate {key!r} is not a covered target")
            candidates.append(
                WeightedCounterfactualCandidate(
                    sample_id=sample_id,
                    deletion=deletion,
                    miss_probability=estimate.clipped_probability,
                    weight=estimate.sampling_weight,
                    max_odds=propensity.max_odds,
                )
            )
    if seen_target_keys != legal_estimate_keys:
        missing = sorted(legal_estimate_keys - seen_target_keys)
        extra = sorted(seen_target_keys - legal_estimate_keys)
        raise ValueError(
            "legal deletion catalog and propensity universe differ: "
            f"missing_deletions={missing!r}, extra_deletions={extra!r}"
        )
    return CUREInterventionCatalog(
        candidates=tuple(sorted(candidates, key=lambda item: item.key)),
        eligible_keys=tuple(
            sorted(
                (item.sample_id, item.gt_id, item.role)
                for item in descriptor_items
            )
        ),
        protocol=protocol,
        frozen_output_fingerprints=tuple(
            (item.sample_id, item.frozen_output_fingerprint)
            for item in sample_catalogs
        ),
        source_fingerprints=tuple(
            (item.sample_id, item.source_fingerprint) for item in sample_catalogs
        ),
        gt_fingerprints=tuple(
            (item.sample_id, item.gt_fingerprint) for item in sample_catalogs
        ),
        propensity_fingerprint=propensity.fingerprint,
        _seal=_INTERVENTION_CATALOG_SEAL,
    )


def choose_weighted_candidate(
    catalog: CUREInterventionCatalog,
    *,
    epoch: int,
    step: int,
    draw_index: int,
    global_seed: int,
) -> WeightedCounterfactualCandidate:
    """Draw reproducibly from the global odds-weighted candidate catalog."""

    if not isinstance(catalog, CUREInterventionCatalog):
        raise TypeError("catalog must be CUREInterventionCatalog")
    catalog.validate_receipt()
    return _choose_weighted_candidate_from_bound_catalog(
        catalog,
        epoch=epoch,
        step=step,
        draw_index=draw_index,
        global_seed=global_seed,
    )


def _choose_weighted_candidate_from_bound_catalog(
    catalog: CUREInterventionCatalog,
    *,
    epoch: int,
    step: int,
    draw_index: int,
    global_seed: int,
) -> WeightedCounterfactualCandidate:
    """Fast draw after a state-pool boundary validated the catalog receipt."""

    for name, value in (("epoch", epoch), ("step", step), ("draw_index", draw_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    candidates = catalog.candidates
    total = math.fsum(item.weight for item in candidates)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("catalog has invalid total weight")
    random_bits = stable_hash(
        "cure-propensity-draw", epoch, step, draw_index, global_seed
    )
    position = (random_bits / float(1 << 64)) * total
    cumulative = 0.0
    for item in candidates:
        cumulative += item.weight
        if position < cumulative:
            return item
    return candidates[-1]


__all__ = [
    "bind_weighted_candidates",
    "choose_weighted_candidate",
    "cross_fit_miss_propensity",
    "miss_odds",
]
