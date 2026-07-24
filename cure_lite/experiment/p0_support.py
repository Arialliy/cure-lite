"""D_R-only common-support and separability diagnostics for P0-B/P0-C."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot, log, log1p, sqrt
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from ..cache.schema import stable_fingerprint
from ..decoder import project_occupancy_to_feature_grid
from ..instances import instances_from_binary_mask
from ..splits import SplitManifest
from .cache_pipeline import LoadedDRCacheBundle
from .p0_protocol import P0OverlapConfig, P0SeparabilityConfig
from .training_pipeline import PreparedTrainingCatalog


P0_B_SCHEMA = "cure-lite-p0-b-common-support-v1"
P0_C_SCHEMA = "cure-lite-p0-c-separability-v1"
_SCALE_FLOOR_RELATIVE = 1e-12


@dataclass(frozen=True)
class _TargetRecord:
    identity: tuple[str, int, int | None]
    sample_id: str
    group_id: str
    role: str
    hand: Tensor
    joint_feature_raw: Tensor
    joint_occupancy_raw: Tensor


@dataclass(frozen=True)
class _FeatureProjector:
    raw_median: Tensor
    raw_scale: Tensor
    raw_constant: Tensor
    raw_maxdev_fallback: Tensor
    pca_mean: Tensor
    basis: Tensor
    singular_values: Tensor


def _dilate(mask: Tensor, radius: int) -> Tensor:
    source = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
    return (
        F.max_pool2d(
            source.to(torch.float32).unsqueeze(0).unsqueeze(0),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )[0, 0]
        > 0
    )


def _background_ring(
    gt_mask: Tensor,
    gt_labels: Tensor,
    valid_mask: Tensor,
    *,
    inner: int,
    outer: int,
) -> Tensor:
    ring = _dilate(gt_mask, outer) & ~_dilate(gt_mask, inner)
    ring &= torch.as_tensor(gt_labels, device="cpu") == 0
    ring &= torch.as_tensor(valid_mask, dtype=torch.bool, device="cpu")
    if not torch.any(ring):
        raise RuntimeError("fixed target background ring is empty")
    return ring.contiguous()


def _feature_moments(feature: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, float]:
    source = torch.as_tensor(feature, dtype=torch.float64, device="cpu")
    if source.ndim != 4 or source.shape[0] != 1:
        raise ValueError("feature must have shape [1,C,h,w]")
    target = torch.as_tensor(mask, dtype=torch.float64, device="cpu")
    weights = F.adaptive_avg_pool2d(
        target.unsqueeze(0).unsqueeze(0),
        tuple(int(value) for value in source.shape[-2:]),
    )[0, 0]
    total = weights.sum()
    if float(total) <= 0.0:
        raise RuntimeError("target has no feature-grid support")
    cells = source[0]
    mean = (cells * weights.unsqueeze(0)).sum(dim=(1, 2)) / total
    variance = (
        (cells - mean[:, None, None]).square() * weights.unsqueeze(0)
    ).sum(dim=(1, 2)) / total
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    rms = float(
        torch.sqrt(
            (cells.square() * weights.unsqueeze(0)).sum()
            / (total * cells.shape[0])
        )
    )
    return mean, std, rms


def _clipped_logit(value: float, epsilon: float) -> float:
    clipped = max(epsilon, min(1.0 - epsilon, float(value)))
    return log(clipped / (1.0 - clipped))


def _mask_mean(value: Tensor, mask: Tensor) -> float:
    selected = torch.as_tensor(value, dtype=torch.float64, device="cpu")[
        torch.as_tensor(mask, dtype=torch.bool, device="cpu")
    ]
    if not selected.numel():
        raise RuntimeError("masked mean has empty support")
    return float(selected.mean())


def _nearest_component_centroid_distance(
    gt_centroid: tuple[float, float],
    occupancy: Tensor,
) -> float:
    components = instances_from_binary_mask(
        occupancy,
        connectivity=8,
        min_area=1,
    )
    height, width = occupancy.shape
    diagonal = hypot(height, width)
    if not components.instances:
        return 1.0
    return min(
        hypot(
            gt_centroid[0] - component.centroid[0],
            gt_centroid[1] - component.centroid[1],
        )
        for component in components.instances
    ) / diagonal


def _occupancy_patch(
    occupancy: Tensor,
    centroid: tuple[float, float],
    feature_size: tuple[int, int],
    radius: int,
) -> Tensor:
    source = torch.as_tensor(occupancy, dtype=torch.bool, device="cpu")
    height, width = source.shape
    feature_height, feature_width = feature_size
    projected = project_occupancy_to_feature_grid(
        source.unsqueeze(0).unsqueeze(0),
        feature_size,
    )[0, 0].to(torch.float64)
    cy = int(round((centroid[0] + 0.5) * feature_height / height - 0.5))
    cx = int(round((centroid[1] + 0.5) * feature_width / width - 0.5))
    cy = min(feature_height - 1, max(0, cy))
    cx = min(feature_width - 1, max(0, cx))
    padded = F.pad(projected, (radius, radius, radius, radius))
    cy += radius
    cx += radius
    patch = padded[
        cy - radius : cy + radius + 1,
        cx - radius : cx + radius + 1,
    ]
    if patch.shape != (2 * radius + 1, 2 * radius + 1):
        raise AssertionError("occupancy patch shape is not fixed")
    return torch.cat((patch.reshape(-1), projected.mean().reshape(1)))


def _target_record(
    *,
    sample_id: str,
    group_id: str,
    role: str,
    gt_id: int,
    pred_id: int | None,
    gt_mask: Tensor,
    supervision_mask: Tensor,
    conditioning_occupancy: Tensor,
    probability: Tensor,
    feature: Tensor,
    gt_labels: Tensor,
    valid_mask: Tensor,
    config: P0OverlapConfig,
) -> _TargetRecord:
    gt = torch.as_tensor(gt_mask, dtype=torch.bool, device="cpu")
    supervision = torch.as_tensor(
        supervision_mask, dtype=torch.bool, device="cpu"
    )
    occupancy = torch.as_tensor(
        conditioning_occupancy, dtype=torch.bool, device="cpu"
    )
    if not torch.any(gt) or not torch.any(supervision):
        raise RuntimeError("P0 target and supervision masks must be nonempty")
    if torch.any(supervision & ~gt):
        raise RuntimeError("P0 supervision is not contained in its GT target")
    ring = _background_ring(
        gt,
        gt_labels,
        valid_mask,
        inner=config.ring_inner_radius,
        outer=config.ring_outer_radius,
    )
    target_mean, target_std, target_rms = _feature_moments(
        feature,
        supervision,
    )
    ring_mean, ring_std, _ = _feature_moments(feature, ring)
    feature64 = torch.as_tensor(feature, dtype=torch.float64, device="cpu")[0]
    global_mean = feature64.mean(dim=(1, 2))
    global_std = feature64.std(dim=(1, 2), unbiased=False)
    coordinates = torch.nonzero(gt, as_tuple=False)
    area = int(coordinates.shape[0])
    supervision_area = int(torch.count_nonzero(supervision))
    ymin = int(coordinates[:, 0].min())
    xmin = int(coordinates[:, 1].min())
    ymax = int(coordinates[:, 0].max()) + 1
    xmax = int(coordinates[:, 1].max()) + 1
    height, width = gt.shape
    centroid = (
        float(coordinates[:, 0].to(torch.float64).mean()),
        float(coordinates[:, 1].to(torch.float64).mean()),
    )
    border_distance = min(ymin, xmin, height - ymax, width - xmax) / max(
        height, width
    )
    base_gt_mean = _mask_mean(probability, gt)
    base_ring_mean = _mask_mean(probability, ring)
    conditioning_gt_fraction = float(
        torch.count_nonzero(occupancy & gt) / area
    )
    conditioning_ring_fraction = float(
        torch.count_nonzero(occupancy & ring)
        / int(torch.count_nonzero(ring))
    )
    hand_values: dict[str, float] = {
        "log1p_gt_area": log1p(area),
        "log1p_supervision_area": log1p(supervision_area),
        "supervision_fraction": supervision_area / area,
        "log_gt_aspect_ratio": log((ymax - ymin) / (xmax - xmin)),
        "border_distance_normalized": border_distance,
        "clipped_logit_base_gt_mean": _clipped_logit(
            base_gt_mean,
            config.probability_clip,
        ),
        "clipped_logit_base_ring_mean": _clipped_logit(
            base_ring_mean,
            config.probability_clip,
        ),
        "log1p_feature_target_rms": log1p(target_rms),
        "feature_target_ring_l2_per_sqrt_channel": float(
            torch.linalg.vector_norm(target_mean - ring_mean)
            / sqrt(target_mean.numel())
        ),
        "conditioning_gt_occupancy_fraction": conditioning_gt_fraction,
        "conditioning_ring_occupancy_fraction": conditioning_ring_fraction,
        "nearest_conditioning_component_centroid_distance_normalized": (
            _nearest_component_centroid_distance(centroid, occupancy)
        ),
    }
    if set(hand_values) != set(config.handcrafted_descriptor_fields):
        raise RuntimeError("handcrafted descriptor fields differ from P0 freeze")
    hand = torch.tensor(
        [hand_values[field] for field in config.handcrafted_descriptor_fields],
        dtype=torch.float64,
    )
    occupancy_patch = _occupancy_patch(
        occupancy,
        centroid,
        tuple(int(value) for value in feature.shape[-2:]),
        config.joint_occupancy_patch_radius,
    )
    joint_feature_raw = torch.cat(
        (
            target_mean,
            target_std,
            ring_mean,
            ring_std,
            global_mean,
            global_std,
        )
    )
    if (
        not torch.isfinite(hand).all()
        or not torch.isfinite(joint_feature_raw).all()
        or not torch.isfinite(occupancy_patch).all()
    ):
        raise RuntimeError("P0 target representation contains non-finite values")
    return _TargetRecord(
        identity=(sample_id, gt_id, pred_id),
        sample_id=sample_id,
        group_id=group_id,
        role=role,
        hand=hand,
        joint_feature_raw=joint_feature_raw,
        joint_occupancy_raw=occupancy_patch,
    )


def _extract_targets(
    bundle: LoadedDRCacheBundle,
    catalog: PreparedTrainingCatalog,
    manifest: SplitManifest,
    config: P0OverlapConfig,
) -> tuple[tuple[_TargetRecord, ...], tuple[_TargetRecord, ...], dict[str, object]]:
    group_by_sample = {
        record.sample_id: record.group_id
        for record in manifest.records_for("D_R")
    }
    if set(group_by_sample) != set(catalog.source_ids):
        raise RuntimeError("P0 manifest groups differ from D_R catalog membership")
    row_by_id = {row.sample_id: row for row in bundle.rows}
    factual: list[_TargetRecord] = []
    legal: list[_TargetRecord] = []
    unreachable: list[dict[str, object]] = []
    for entry in catalog.entries:
        row = row_by_id[entry.sample_id]
        state = row.state
        gt = entry.gt
        for gt_id, example in zip(
            entry.reachable_gt_ids,
            entry.factual_examples,
            strict=True,
        ):
            factual.append(
                _target_record(
                    sample_id=entry.sample_id,
                    group_id=group_by_sample[entry.sample_id],
                    role="factual",
                    gt_id=gt_id,
                    pred_id=None,
                    gt_mask=gt.by_id(gt_id).mask,
                    supervision_mask=example.supervision.target[0] > 0,
                    conditioning_occupancy=example.supervision.occupancy[0],
                    probability=row.base_output.probability[0, 0],
                    feature=row.base_output.feature,
                    gt_labels=state.gt_labels,
                    valid_mask=state.image_valid_mask,
                    config=config,
                )
            )
        unreachable.extend(
            {
                "sample_id": entry.sample_id,
                "group_id": group_by_sample[entry.sample_id],
                "gt_id": gt_id,
            }
            for gt_id in sorted(
                set(entry.real_miss_ids) - set(entry.reachable_gt_ids)
            )
        )
        for candidate, example in zip(
            entry.decoder_visible_legal_candidates,
            entry.synthetic_examples,
            strict=True,
        ):
            legal.append(
                _target_record(
                    sample_id=entry.sample_id,
                    group_id=group_by_sample[entry.sample_id],
                    role="legal",
                    gt_id=candidate.gt_id,
                    pred_id=candidate.pred_id,
                    gt_mask=gt.by_id(candidate.gt_id).mask,
                    supervision_mask=example.supervision.target[0] > 0,
                    conditioning_occupancy=example.supervision.occupancy[0],
                    probability=row.base_output.probability[0, 0],
                    feature=row.base_output.feature,
                    gt_labels=state.gt_labels,
                    valid_mask=state.image_valid_mask,
                    config=config,
                )
            )
    factual.sort(key=lambda item: item.identity)
    legal.sort(key=lambda item: item.identity)
    if len(factual) != catalog.support_summary.reachable_miss_targets:
        raise RuntimeError("P0 factual target count differs from prepared catalog")
    if len(legal) != catalog.support_summary.decoder_visible_legal_candidates:
        raise RuntimeError("P0 legal target count differs from prepared catalog")
    return (
        tuple(factual),
        tuple(legal),
        {
            "reachable_factual_targets": len(factual),
            "reachable_factual_groups": len({item.group_id for item in factual}),
            "unreachable_factual_targets": len(unreachable),
            "unreachable_factual_identities": unreachable,
            "decoder_visible_legal_targets": len(legal),
            "decoder_visible_legal_groups": len(
                {item.group_id for item in legal}
            ),
            "groups_with_both_roles": len(
                {item.group_id for item in factual}
                & {item.group_id for item in legal}
            ),
        },
    )


def _fit_pca(
    values: Tensor,
    components: int,
) -> tuple[Tensor, Tensor, Tensor]:
    source = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    mean = source.mean(dim=0)
    centered = source - mean
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = int(torch.count_nonzero(singular > 1e-12))
    if rank < components:
        raise RuntimeError(
            f"joint representation rank {rank} is below frozen dimension {components}"
        )
    basis = vh[:components].clone()
    for index in range(components):
        pivot = int(torch.argmax(torch.abs(basis[index])))
        if float(basis[index, pivot]) < 0.0:
            basis[index] *= -1.0
    return mean, basis, singular[:components]


def _project_pca(values: Tensor, mean: Tensor, basis: Tensor) -> Tensor:
    return (torch.as_tensor(values, dtype=torch.float64) - mean) @ basis.T


def _robust_scale_fit(
    values: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    source = torch.as_tensor(values, dtype=torch.float64)
    median = torch.median(source, dim=0).values
    absolute_deviation = torch.abs(source - median)
    mad = torch.median(absolute_deviation, dim=0).values
    maxdev = torch.max(absolute_deviation, dim=0).values
    floor = (
        torch.maximum(torch.ones_like(median), torch.abs(median))
        * _SCALE_FLOOR_RELATIVE
    )
    use_mad = mad > floor
    use_maxdev = (~use_mad) & (maxdev > floor)
    constant = (~use_mad) & (~use_maxdev)
    scale = torch.where(
        use_mad,
        mad,
        torch.where(use_maxdev, maxdev, floor),
    )
    if torch.any(scale <= 0.0) or not torch.isfinite(scale).all():
        raise RuntimeError("robust scale is non-positive or non-finite")
    return median, scale, constant, use_maxdev


def _robust_scale(
    values: Tensor,
    median: Tensor,
    scale: Tensor,
) -> Tensor:
    return (torch.as_tensor(values, dtype=torch.float64) - median) / scale


def _fit_feature_projector(
    values: Tensor,
    components: int,
) -> _FeatureProjector:
    median, scale, constant, maxdev_fallback = _robust_scale_fit(values)
    standardized = _robust_scale(values, median, scale)
    pca_mean, basis, singular = _fit_pca(standardized, components)
    return _FeatureProjector(
        raw_median=median,
        raw_scale=scale,
        raw_constant=constant,
        raw_maxdev_fallback=maxdev_fallback,
        pca_mean=pca_mean,
        basis=basis,
        singular_values=singular,
    )


def _project_feature(
    values: Tensor,
    projector: _FeatureProjector,
) -> Tensor:
    standardized = _robust_scale(
        values,
        projector.raw_median,
        projector.raw_scale,
    )
    centered = standardized - projector.pca_mean
    scores = centered @ projector.basis.T
    reconstruction = scores @ projector.basis
    residual = torch.linalg.vector_norm(
        centered - reconstruction,
        dim=1,
        keepdim=True,
    ) / sqrt(centered.shape[1])
    return torch.cat((scores, residual), dim=1)


def _higher_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile values cannot be empty")
    index = max(0, ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _group_kth_distance(
    query: Tensor,
    query_group: str,
    references: Tensor,
    reference_groups: Sequence[str],
    k: int,
) -> float:
    by_group: dict[str, float] = {}
    distances = torch.linalg.vector_norm(references - query, dim=1)
    for distance, group in zip(distances.tolist(), reference_groups, strict=True):
        if group == query_group:
            continue
        by_group[group] = min(by_group.get(group, float("inf")), float(distance))
    if len(by_group) < k:
        raise RuntimeError("fewer than k source-disjoint legal groups are available")
    return sorted(by_group.values())[k - 1]


def _coverage_receipt(
    factual: Sequence[_TargetRecord],
    legal: Sequence[_TargetRecord],
    factual_values: Tensor,
    legal_values: Tensor,
    config: P0OverlapConfig,
) -> dict[str, object]:
    median, scale, constant, maxdev_fallback = _robust_scale_fit(legal_values)
    factual_scaled = _robust_scale(factual_values, median, scale)
    legal_scaled = _robust_scale(legal_values, median, scale)
    legal_groups = [item.group_id for item in legal]
    reference_distances = [
        _group_kth_distance(
            legal_scaled[index],
            legal[index].group_id,
            legal_scaled,
            legal_groups,
            config.knn_k,
        )
        for index in range(len(legal))
    ]
    radius = _higher_quantile(
        reference_distances,
        config.legal_reference_quantile,
    )
    factual_rows: list[dict[str, object]] = []
    covered = 0
    for index, item in enumerate(factual):
        distance = _group_kth_distance(
            factual_scaled[index],
            item.group_id,
            legal_scaled,
            legal_groups,
            config.knn_k,
        )
        inside = distance <= radius
        covered += int(inside)
        factual_rows.append(
            {
                "identity": list(item.identity),
                "group_id": item.group_id,
                "source_disjoint_kth_distance": distance,
                "covered": inside,
            }
        )
    required = ceil(config.coverage_minimum * len(factual))
    return {
        "k": config.knn_k,
        "reference_quantile": config.legal_reference_quantile,
        "reference_radius": radius,
        "dimensions": int(legal_values.shape[1]),
        "maxdev_fallback_dimensions": [
            index
            for index, flag in enumerate(maxdev_fallback.tolist())
            if flag
        ],
        "constant_floor_dimensions": [
            index for index, flag in enumerate(constant.tolist()) if flag
        ],
        "legal_median": [float(value) for value in median.tolist()],
        "legal_scale": [float(value) for value in scale.tolist()],
        "factual_total": len(factual),
        "covered_factual_targets": covered,
        "required_covered_factual_targets": required,
        "coverage": covered / len(factual),
        "pass": covered >= required,
        "factual_targets": factual_rows,
        "legal_reference_kth_distance_summary": {
            "minimum": min(reference_distances),
            "median": _higher_quantile(reference_distances, 0.5),
            "maximum": max(reference_distances),
        },
    }


def _fold_assignment(
    records: Sequence[_TargetRecord],
    folds: int,
) -> dict[str, int]:
    groups = sorted({item.group_id for item in records})
    counts: dict[str, tuple[int, int]] = {}
    for group in groups:
        counts[group] = (
            sum(item.group_id == group and item.role == "factual" for item in records),
            sum(item.group_id == group and item.role == "legal" for item in records),
        )
    ordered = sorted(
        groups,
        key=lambda group: (
            -int(counts[group][0] > 0),
            -counts[group][0],
            -counts[group][1],
            group,
        ),
    )
    totals = [[0, 0, 0] for _ in range(folds)]
    assignment: dict[str, int] = {}
    for group in ordered:
        factual_count, legal_count = counts[group]
        fold = min(
            range(folds),
            key=lambda index: (
                totals[index][0] if factual_count else totals[index][1],
                totals[index][1],
                totals[index][2],
                index,
            ),
        )
        assignment[group] = fold
        totals[fold][0] += factual_count
        totals[fold][1] += legal_count
        totals[fold][2] += factual_count + legal_count
    return assignment


def _fit_logistic_irls(
    values: Tensor,
    labels: Tensor,
    groups: Sequence[str],
    config: P0SeparabilityConfig,
) -> tuple[Tensor, dict[str, object]]:
    x = torch.as_tensor(values, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    design = torch.cat((torch.ones((x.shape[0], 1), dtype=x.dtype), x), dim=1)
    weights = torch.empty_like(y)
    for label in (0, 1):
        label_groups = sorted(
            {
                group
                for group, value in zip(groups, y.tolist(), strict=True)
                if int(value) == label
            }
        )
        if not label_groups:
            raise RuntimeError("classifier fold lacks one target role")
        for group in label_groups:
            indices = [
                index
                for index, (candidate, value) in enumerate(
                    zip(groups, y.tolist(), strict=True)
                )
                if candidate == group and int(value) == label
            ]
            mass = 0.5 / len(label_groups) / len(indices)
            weights[indices] = mass
    beta = torch.zeros(design.shape[1], dtype=torch.float64)
    penalty = torch.eye(design.shape[1], dtype=torch.float64)
    penalty[0, 0] = 0.0
    converged = False
    delta_norm = float("inf")
    iterations = 0
    for iteration in range(1, config.classifier_max_iterations + 1):
        probability = torch.sigmoid(design @ beta)
        gradient = design.T @ (weights * (probability - y))
        gradient += config.classifier_l2 * (penalty @ beta)
        curvature = weights * probability * (1.0 - probability)
        hessian = design.T @ (curvature[:, None] * design)
        hessian += config.classifier_l2 * penalty
        hessian += torch.eye(hessian.shape[0], dtype=hessian.dtype) * 1e-12
        try:
            delta = torch.linalg.solve(hessian, gradient)
        except RuntimeError:
            delta = torch.linalg.pinv(hessian) @ gradient
        if not torch.isfinite(delta).all():
            raise RuntimeError("grouped logistic IRLS produced a non-finite step")
        beta -= delta
        iterations = iteration
        delta_norm = float(torch.linalg.vector_norm(delta))
        if delta_norm <= config.classifier_tolerance:
            converged = True
            break
    final_probability = torch.sigmoid(design @ beta)
    final_gradient = design.T @ (weights * (final_probability - y))
    final_gradient += config.classifier_l2 * (penalty @ beta)
    gradient_norm = float(torch.linalg.vector_norm(final_gradient))
    if not converged:
        raise RuntimeError(
            "grouped logistic IRLS did not converge within the frozen limit"
        )
    return beta, {
        "converged": True,
        "iterations": iterations,
        "maximum_iterations": config.classifier_max_iterations,
        "delta_norm": delta_norm,
        "gradient_norm": gradient_norm,
        "tolerance": config.classifier_tolerance,
    }


def _group_balanced_weights(
    labels: Sequence[int],
    groups: Sequence[str],
) -> list[float]:
    result = [0.0] * len(labels)
    for label in (0, 1):
        label_groups = sorted(
            {
                group
                for group, value in zip(groups, labels, strict=True)
                if value == label
            }
        )
        if not label_groups:
            continue
        for group in label_groups:
            indices = [
                index
                for index, (candidate, value) in enumerate(
                    zip(groups, labels, strict=True)
                )
                if candidate == group and value == label
            ]
            for index in indices:
                result[index] = 1.0 / len(label_groups) / len(indices)
    return result


def _weighted_auc(
    scores: Sequence[float],
    labels: Sequence[int],
    weights: Sequence[float],
) -> float:
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    if not positives or not negatives:
        raise RuntimeError("AUC requires both target roles")
    numerator = 0.0
    denominator = 0.0
    for positive in positives:
        for negative in negatives:
            mass = weights[positive] * weights[negative]
            denominator += mass
            numerator += mass * (
                1.0
                if scores[positive] > scores[negative]
                else 0.5
                if scores[positive] == scores[negative]
                else 0.0
            )
    return numerator / denominator


def _oof_auc(
    records: Sequence[_TargetRecord],
    *,
    space: str,
    overlap: P0OverlapConfig,
    config: P0SeparabilityConfig,
) -> dict[str, object]:
    assignment = _fold_assignment(records, config.folds)
    scores = [float("nan")] * len(records)
    labels = [1 if item.role == "factual" else 0 for item in records]
    groups = [item.group_id for item in records]
    fold_rows: list[dict[str, object]] = []
    for fold in range(config.folds):
        train = [
            index for index, group in enumerate(groups) if assignment[group] != fold
        ]
        test = [
            index for index, group in enumerate(groups) if assignment[group] == fold
        ]
        if not train or not test:
            raise RuntimeError("grouped classifier produced an empty fold")
        if space == "handcrafted":
            train_raw = torch.stack([records[index].hand for index in train])
            test_raw = torch.stack([records[index].hand for index in test])
        elif space == "joint":
            legal_feature_train = torch.stack(
                [
                    records[index].joint_feature_raw
                    for index in train
                    if records[index].role == "legal"
                ]
            )
            feature_projector = _fit_feature_projector(
                legal_feature_train,
                overlap.joint_feature_components,
            )
            train_raw = torch.cat(
                (
                    _project_feature(
                        torch.stack(
                            [
                                records[index].joint_feature_raw
                                for index in train
                            ]
                        ),
                        feature_projector,
                    ),
                    torch.stack(
                        [
                            records[index].joint_occupancy_raw
                            for index in train
                        ]
                    ),
                ),
                dim=1,
            )
            test_raw = torch.cat(
                (
                    _project_feature(
                        torch.stack(
                            [
                                records[index].joint_feature_raw
                                for index in test
                            ]
                        ),
                        feature_projector,
                    ),
                    torch.stack(
                        [
                            records[index].joint_occupancy_raw
                            for index in test
                        ]
                    ),
                ),
                dim=1,
            )
        else:
            raise ValueError("unknown P0 representation space")
        median, scale, constant, maxdev_fallback = _robust_scale_fit(
            train_raw
        )
        train_values = _robust_scale(train_raw, median, scale)
        test_values = _robust_scale(test_raw, median, scale)
        beta, fit_receipt = _fit_logistic_irls(
            train_values,
            torch.tensor([labels[index] for index in train], dtype=torch.float64),
            [groups[index] for index in train],
            config,
        )
        probabilities = torch.sigmoid(
            torch.cat(
                (
                    torch.ones((test_values.shape[0], 1), dtype=torch.float64),
                    test_values,
                ),
                dim=1,
            )
            @ beta
        )
        for index, score in zip(test, probabilities.tolist(), strict=True):
            scores[index] = float(score)
        fold_rows.append(
            {
                "fold": fold,
                "train_groups": sorted(
                    {groups[index] for index in train}
                ),
                "test_groups": sorted({groups[index] for index in test}),
                "dimensions": int(train_raw.shape[1]),
                "maxdev_fallback_dimensions": [
                    index
                    for index, flag in enumerate(maxdev_fallback.tolist())
                    if flag
                ],
                "constant_floor_dimensions": [
                    index
                    for index, flag in enumerate(constant.tolist())
                    if flag
                ],
                "classifier_fit": fit_receipt,
            }
        )
    if any(not torch.isfinite(torch.tensor(value)) for value in scores):
        raise RuntimeError("grouped classifier left an OOF score undefined")
    weights = _group_balanced_weights(labels, groups)
    auc = _weighted_auc(scores, labels, weights)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.bootstrap_seed)
    unique_groups = sorted(set(groups))
    by_group = {
        group: [index for index, candidate in enumerate(groups) if candidate == group]
        for group in unique_groups
    }
    bootstrap: list[float] = []
    skipped = 0
    for _ in range(config.bootstrap_replicates):
        sampled = torch.randint(
            len(unique_groups),
            (len(unique_groups),),
            generator=generator,
        ).tolist()
        selected_indices: list[int] = []
        selected_groups: list[str] = []
        for occurrence, group_index in enumerate(sampled):
            group = unique_groups[group_index]
            selected_indices.extend(by_group[group])
            selected_groups.extend(
                [f"{group}#{occurrence}"] * len(by_group[group])
            )
        selected_labels = [labels[index] for index in selected_indices]
        if len(set(selected_labels)) < 2:
            skipped += 1
            continue
        selected_scores = [scores[index] for index in selected_indices]
        selected_weights = _group_balanced_weights(
            selected_labels,
            selected_groups,
        )
        bootstrap.append(
            _weighted_auc(selected_scores, selected_labels, selected_weights)
        )
    if not bootstrap:
        raise RuntimeError("all grouped AUC bootstrap replicates were uninformative")
    lower, upper = config.bootstrap_interval
    return {
        "space": space,
        "grouped_oof_auc": auc,
        "auc_maximum": config.auc_maximum,
        "gate_rule": config.auc_gate_rule,
        "pass": auc <= config.auc_maximum,
        "bootstrap": {
            "interpretation": config.bootstrap_interpretation,
            "requested_replicates": config.bootstrap_replicates,
            "valid_replicates": len(bootstrap),
            "skipped_replicates": skipped,
            "interval_probability": [lower, upper],
            "lower": _higher_quantile(bootstrap, lower),
            "upper": _higher_quantile(bootstrap, upper),
        },
        "folds": fold_rows,
        "oof_predictions": [
            {
                "identity": list(item.identity),
                "group_id": item.group_id,
                "role": item.role,
                "score_factual": score,
            }
            for item, score in zip(records, scores, strict=True)
        ],
    }


def _kernel(
    values: Tensor,
    bandwidth: float,
    scales: Sequence[float],
) -> Tensor:
    distances2 = torch.cdist(values, values).square()
    kernels = [
        torch.exp(-distances2 / (2.0 * (scale * bandwidth) ** 2))
        for scale in scales
    ]
    return sum(kernels) / len(kernels)


def _group_kernel(
    target_kernel: Tensor,
    records: Sequence[_TargetRecord],
    groups: Sequence[str],
) -> Tensor:
    members = {
        group: [
            index
            for index, record in enumerate(records)
            if record.group_id == group
        ]
        for group in groups
    }
    result = torch.empty(
        (len(groups), len(groups)),
        dtype=torch.float64,
    )
    for first, first_group in enumerate(groups):
        first_index = torch.tensor(members[first_group], dtype=torch.int64)
        for second in range(first, len(groups)):
            second_index = torch.tensor(
                members[groups[second]],
                dtype=torch.int64,
            )
            value = target_kernel[first_index][:, second_index].mean()
            result[first, second] = value
            result[second, first] = value
    return result


def _group_mmd_u(
    kernel: Tensor,
    first: Sequence[int],
    second: Sequence[int],
) -> float:
    if len(first) < 2 or len(second) < 2:
        raise RuntimeError("group MMD requires at least two groups per side")
    if set(first) & set(second):
        raise RuntimeError("group MMD sides must be source-disjoint")
    first_index = torch.tensor(first, dtype=torch.int64)
    second_index = torch.tensor(second, dtype=torch.int64)
    k_xx = kernel[first_index][:, first_index]
    k_yy = kernel[second_index][:, second_index]
    k_xy = kernel[first_index][:, second_index]
    first_within = (
        k_xx.sum() - torch.diagonal(k_xx).sum()
    ) / (len(first) * (len(first) - 1))
    second_within = (
        k_yy.sum() - torch.diagonal(k_yy).sum()
    ) / (len(second) * (len(second) - 1))
    value = (
        first_within
        + second_within
        - 2.0 * k_xy.mean()
    )
    return float(value)


def _mmd_receipt(
    factual: Sequence[_TargetRecord],
    legal: Sequence[_TargetRecord],
    factual_values: Tensor,
    legal_values: Tensor,
    *,
    space: str,
    config: P0SeparabilityConfig,
) -> dict[str, object]:
    factual_groups = sorted({item.group_id for item in factual})
    factual_group_set = set(factual_groups)
    legal_exclusive_indices = [
        index
        for index, item in enumerate(legal)
        if item.group_id not in factual_group_set
    ]
    legal_exclusive = tuple(legal[index] for index in legal_exclusive_indices)
    legal_exclusive_values = legal_values[legal_exclusive_indices]
    legal_groups = sorted({item.group_id for item in legal_exclusive})
    if len(factual_groups) < 2:
        raise RuntimeError("MMD has fewer than two factual groups")
    if len(legal_groups) < 2 * len(factual_groups):
        raise RuntimeError(
            "MMD legal-exclusive groups cannot form matched 24-vs-rest splits"
        )

    median, scale, constant, maxdev_fallback = _robust_scale_fit(
        legal_exclusive_values
    )
    factual_scaled = _robust_scale(factual_values, median, scale)
    legal_scaled = _robust_scale(legal_exclusive_values, median, scale)
    legal_distances: list[float] = []
    for first in range(len(legal_exclusive)):
        for second in range(first + 1, len(legal_exclusive)):
            if (
                legal_exclusive[first].group_id
                == legal_exclusive[second].group_id
            ):
                continue
            distance = float(
                torch.linalg.vector_norm(
                    legal_scaled[first] - legal_scaled[second]
                )
            )
            if distance > 0.0:
                legal_distances.append(distance)
    if not legal_distances:
        raise RuntimeError("MMD has no positive source-disjoint legal distance")
    bandwidth = _higher_quantile(legal_distances, 0.5)
    combined = torch.cat((factual_scaled, legal_scaled), dim=0)
    combined_records = (*factual, *legal_exclusive)
    group_order = (*factual_groups, *legal_groups)
    target_kernel = _kernel(
        combined,
        bandwidth,
        config.mmd_kernel_scales,
    )
    group_kernel = _group_kernel(
        target_kernel,
        combined_records,
        group_order,
    )
    factual_group_count = len(factual_groups)
    factual_group_indices = list(range(factual_group_count))
    legal_offset = factual_group_count
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.mmd_reference_seed)
    observed_values: list[float] = []
    reference_values: list[float] = []
    partition_fingerprints: list[str] = []
    for _ in range(config.mmd_reference_replicates):
        order = torch.randperm(len(legal_groups), generator=generator).tolist()
        first = [
            legal_offset + index
            for index in order[:factual_group_count]
        ]
        second = [
            legal_offset + index
            for index in order[factual_group_count:]
        ]
        observed_values.append(
            _group_mmd_u(
                group_kernel,
                factual_group_indices,
                second,
            )
        )
        reference_values.append(
            _group_mmd_u(group_kernel, first, second)
        )
        partition_fingerprints.append(
            stable_fingerprint(
                {
                    "first": [group_order[index] for index in first],
                    "second": [group_order[index] for index in second],
                }
            )
        )
    observed = _higher_quantile(
        observed_values,
        config.mmd_observed_summary_quantile,
    )
    q95 = _higher_quantile(
        reference_values,
        config.mmd_reference_quantile,
    )
    return {
        "space": space,
        "estimator": config.mmd,
        "group_overlap_policy": config.mmd_group_overlap_policy,
        "bandwidth_rule": config.mmd_bandwidth_rule,
        "bandwidth": bandwidth,
        "kernel_scales": list(config.mmd_kernel_scales),
        "groups": {
            "factual": len(factual_groups),
            "legal_all": len({item.group_id for item in legal}),
            "overlap_removed_from_legal": len(
                {item.group_id for item in legal} & factual_group_set
            ),
            "legal_exclusive": len(legal_groups),
            "observed_left": len(factual_groups),
            "matched_null_left": len(factual_groups),
            "shared_right": len(legal_groups) - len(factual_groups),
        },
        "standardization": {
            "legal_exclusive_median": [
                float(value) for value in median.tolist()
            ],
            "legal_exclusive_scale": [
                float(value) for value in scale.tolist()
            ],
            "maxdev_fallback_dimensions": [
                index
                for index, flag in enumerate(maxdev_fallback.tolist())
                if flag
            ],
            "constant_floor_dimensions": [
                index for index, flag in enumerate(constant.tolist()) if flag
            ],
        },
        "observed_factual_vs_matched_legal": {
            "replicates": len(observed_values),
            "summary_quantile_probability": (
                config.mmd_observed_summary_quantile
            ),
            "summary_quantile": observed,
            "values": observed_values,
        },
        "legal_vs_legal_reference": {
            "replicates": len(reference_values),
            "median": _higher_quantile(reference_values, 0.5),
            "quantile_probability": config.mmd_reference_quantile,
            "quantile": q95,
            "values": reference_values,
        },
        "partition_sequence_fingerprint": stable_fingerprint(
            partition_fingerprints
        ),
        "observed_minus_reference_quantile": observed - q95,
        "pass": observed <= q95,
    }


def build_p0_b_c_support(
    bundle: LoadedDRCacheBundle,
    catalog: PreparedTrainingCatalog,
    manifest: SplitManifest,
    overlap: P0OverlapConfig,
    separability: P0SeparabilityConfig,
    *,
    formal_eligible: bool,
) -> tuple[dict[str, object], dict[str, object], dict[str, Tensor]]:
    """Build P0-B/C diagnostics, gated sequentially behind P0-A."""

    if not isinstance(formal_eligible, bool):
        raise TypeError("formal_eligible must be bool")

    factual, legal, counts = _extract_targets(
        bundle,
        catalog,
        manifest,
        overlap,
    )
    factual_hand = torch.stack([item.hand for item in factual])
    legal_hand = torch.stack([item.hand for item in legal])
    factual_joint_feature_raw = torch.stack(
        [item.joint_feature_raw for item in factual]
    )
    legal_joint_feature_raw = torch.stack(
        [item.joint_feature_raw for item in legal]
    )
    factual_joint_occupancy_raw = torch.stack(
        [item.joint_occupancy_raw for item in factual]
    )
    legal_joint_occupancy_raw = torch.stack(
        [item.joint_occupancy_raw for item in legal]
    )
    feature_projector = _fit_feature_projector(
        legal_joint_feature_raw,
        overlap.joint_feature_components,
    )
    factual_joint = torch.cat(
        (
            _project_feature(
                factual_joint_feature_raw,
                feature_projector,
            ),
            factual_joint_occupancy_raw,
        ),
        dim=1,
    )
    legal_joint = torch.cat(
        (
            _project_feature(
                legal_joint_feature_raw,
                feature_projector,
            ),
            legal_joint_occupancy_raw,
        ),
        dim=1,
    )
    factual_group_ids = {item.group_id for item in factual}
    mmd_legal_indices = [
        index
        for index, item in enumerate(legal)
        if item.group_id not in factual_group_ids
    ]
    mmd_feature_projector = _fit_feature_projector(
        legal_joint_feature_raw[mmd_legal_indices],
        overlap.joint_feature_components,
    )
    factual_joint_mmd = torch.cat(
        (
            _project_feature(
                factual_joint_feature_raw,
                mmd_feature_projector,
            ),
            factual_joint_occupancy_raw,
        ),
        dim=1,
    )
    legal_joint_mmd = torch.cat(
        (
            _project_feature(
                legal_joint_feature_raw,
                mmd_feature_projector,
            ),
            legal_joint_occupancy_raw,
        ),
        dim=1,
    )

    coverage_hand = _coverage_receipt(
        factual,
        legal,
        factual_hand,
        legal_hand,
        overlap,
    )
    coverage_joint = _coverage_receipt(
        factual,
        legal,
        factual_joint,
        legal_joint,
        overlap,
    )
    p0_b_diagnostic_pass = bool(
        coverage_hand["pass"] and coverage_joint["pass"]
    )
    p0_b_pass = p0_b_diagnostic_pass if formal_eligible else None
    target_catalog = [
        {
            "identity": list(item.identity),
            "group_id": item.group_id,
            "role": item.role,
            "handcrafted_descriptor": [
                float(value) for value in item.hand.tolist()
            ],
            "joint_projection": [
                float(value)
                for value in (
                    factual_joint[index]
                    if item.role == "factual"
                    else legal_joint[index - len(factual)]
                ).tolist()
            ],
        }
        for index, item in enumerate((*factual, *legal))
    ]
    p0_b = {
        "schema_version": P0_B_SCHEMA,
        "split": "D_R",
        "counts": counts,
        "coverage": {
            "handcrafted": coverage_hand,
            "decoder_joint": coverage_joint,
        },
        "joint_projection": {
            "feature_raw_dimensions": int(
                legal_joint_feature_raw.shape[1]
            ),
            "occupancy_raw_dimensions": int(
                legal_joint_occupancy_raw.shape[1]
            ),
            "feature_components": overlap.joint_feature_components,
            "feature_residual": overlap.joint_feature_residual,
            "occupancy_representation": (
                overlap.joint_occupancy_representation
            ),
            "feature_raw_median": [
                float(value)
                for value in feature_projector.raw_median.tolist()
            ],
            "feature_raw_scale": [
                float(value)
                for value in feature_projector.raw_scale.tolist()
            ],
            "feature_raw_maxdev_fallback_dimensions": [
                index
                for index, flag in enumerate(
                    feature_projector.raw_maxdev_fallback.tolist()
                )
                if flag
            ],
            "feature_raw_constant_floor_dimensions": [
                index
                for index, flag in enumerate(
                    feature_projector.raw_constant.tolist()
                )
                if flag
            ],
            "feature_pca_mean": [
                float(value)
                for value in feature_projector.pca_mean.tolist()
            ],
            "feature_basis": [
                [float(value) for value in row]
                for row in feature_projector.basis.tolist()
            ],
            "feature_singular_values": [
                float(value)
                for value in feature_projector.singular_values.tolist()
            ],
            "basis_fingerprint": stable_fingerprint(
                {
                    "feature_raw_median": [
                        float(value)
                        for value in feature_projector.raw_median.tolist()
                    ],
                    "feature_raw_scale": [
                        float(value)
                        for value in feature_projector.raw_scale.tolist()
                    ],
                    "feature_pca_mean": [
                        float(value)
                        for value in feature_projector.pca_mean.tolist()
                    ],
                    "feature_basis": [
                        [float(value) for value in row]
                        for row in feature_projector.basis.tolist()
                    ],
                    "occupancy_representation": (
                        overlap.joint_occupancy_representation
                    ),
                }
            ),
        },
        "target_catalog": target_catalog,
        "diagnostic_status": (
            "pass" if p0_b_diagnostic_pass else "fail"
        ),
        "diagnostic_pass": p0_b_diagnostic_pass,
        "formal_status": (
            "pass"
            if p0_b_diagnostic_pass and formal_eligible
            else "fail"
            if formal_eligible
            else "not_evaluated_due_to_p0_a_failure"
        ),
        "p0_b_pass": p0_b_pass,
        "failure_decision": (
            None
            if p0_b_pass is True
            else "redesign_synthetic_state"
            if formal_eligible
            else "follow_p0_a_failure"
        ),
    }

    records = (*factual, *legal)
    try:
        auc_hand = _oof_auc(
            records,
            space="handcrafted",
            overlap=overlap,
            config=separability,
        )
        auc_joint = _oof_auc(
            records,
            space="joint",
            overlap=overlap,
            config=separability,
        )
        mmd_hand = _mmd_receipt(
            factual,
            legal,
            factual_hand,
            legal_hand,
            space="handcrafted",
            config=separability,
        )
        mmd_joint = _mmd_receipt(
            factual,
            legal,
            factual_joint_mmd,
            legal_joint_mmd,
            space="joint",
            config=separability,
        )
        p0_c_diagnostic_pass: bool | None = bool(
            auc_hand["pass"]
            and auc_joint["pass"]
            and mmd_hand["pass"]
            and mmd_joint["pass"]
        )
        diagnostic_status = "pass" if p0_c_diagnostic_pass else "fail"
        diagnostic_error = None
    except (RuntimeError, ValueError) as error:
        auc_hand = auc_joint = mmd_hand = mmd_joint = None
        p0_c_diagnostic_pass = None
        diagnostic_status = "inconclusive"
        diagnostic_error = str(error)
    p0_c_pass = (
        bool(p0_c_diagnostic_pass)
        if formal_eligible and p0_c_diagnostic_pass is not None
        else False
        if formal_eligible
        else None
    )
    formal_status = (
        diagnostic_status
        if formal_eligible
        else "not_evaluated_due_to_p0_a_failure"
    )
    failure = (
        None
        if p0_c_pass is True
        else "redesign_synthetic_state"
        if formal_eligible
        else "follow_p0_a_failure"
    )
    p0_c = {
        "schema_version": P0_C_SCHEMA,
        "split": "D_R",
        "diagnostic_status": diagnostic_status,
        "diagnostic_pass": p0_c_diagnostic_pass,
        "formal_status": formal_status,
        "grouped_classifier": {
            "handcrafted": auc_hand,
            "decoder_joint": auc_joint,
        },
        "mmd": {
            "handcrafted": mmd_hand,
            "decoder_joint": mmd_joint,
        },
        "mmd_joint_feature_projection": {
            "fit_population": "legal-exclusive-source-groups",
            "fit_targets": len(mmd_legal_indices),
            "fit_groups": len(
                {
                    legal[index].group_id
                    for index in mmd_legal_indices
                }
            ),
            "raw_median": [
                float(value)
                for value in mmd_feature_projector.raw_median.tolist()
            ],
            "raw_scale": [
                float(value)
                for value in mmd_feature_projector.raw_scale.tolist()
            ],
            "pca_mean": [
                float(value)
                for value in mmd_feature_projector.pca_mean.tolist()
            ],
            "basis": [
                [float(value) for value in row]
                for row in mmd_feature_projector.basis.tolist()
            ],
            "singular_values": [
                float(value)
                for value in mmd_feature_projector.singular_values.tolist()
            ],
            "projection_fingerprint": stable_fingerprint(
                {
                    "raw_median": [
                        float(value)
                        for value in mmd_feature_projector.raw_median.tolist()
                    ],
                    "raw_scale": [
                        float(value)
                        for value in mmd_feature_projector.raw_scale.tolist()
                    ],
                    "pca_mean": [
                        float(value)
                        for value in mmd_feature_projector.pca_mean.tolist()
                    ],
                    "basis": [
                        [float(value) for value in row]
                        for row in mmd_feature_projector.basis.tolist()
                    ],
                }
            ),
        },
        "diagnostic_error": diagnostic_error,
        "p0_c_pass": p0_c_pass,
        "failure_decision": failure,
    }
    matrices = {
        "factual_hand": factual_hand,
        "legal_hand": legal_hand,
        "factual_joint": factual_joint,
        "legal_joint": legal_joint,
    }
    return p0_b, p0_c, matrices


__all__ = [
    "P0_B_SCHEMA",
    "P0_C_SCHEMA",
    "build_p0_b_c_support",
]
