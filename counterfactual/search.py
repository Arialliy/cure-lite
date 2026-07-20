"""Minimal legal search over model-consistent CURE input interventions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from ..config import MatchConfig, OccupancyConfig
from ..frozen_base import FrozenBaseAdapter
from ..instances import union_instance_masks
from ..matching import match_components
from ..occupancy import build_occupancy
from ..supervision import build_atomic_intervention_supervision
from ..types import BranchSupervision, Instance, InstanceMap, MatchResult
from .acceptance import (
    AcceptanceConfig,
    AcceptanceDecision,
    assess_legal_intervention,
)
from .contracts import TransformConfig, TransformDiagnostics, TransformSpec
from .transforms import InvalidTransformSupportError, apply_counterfactual_transform


_TRANSFORM_ERRORS = frozenset(
    {"invalid_transform_support", "transform_has_no_effect"}
)
_SEARCH_FAILURES = frozenset(
    {"target_not_matched_before", "no_legal_candidate"}
)


def _config_fingerprint(config: object) -> str:
    payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_fingerprint(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _clone_instance_map(value: InstanceMap) -> InstanceMap:
    return InstanceMap(
        labels=value.labels.clone(),
        instances=tuple(
            Instance(
                instance_id=item.instance_id,
                mask=item.mask.clone(),
                area=item.area,
                bbox=item.bbox,
                centroid=item.centroid,
            )
            for item in value.instances
        ),
    )


def _clone_supervision(value: BranchSupervision) -> BranchSupervision:
    return BranchSupervision(
        occupancy=value.occupancy.clone(),
        target=value.target.clone(),
        valid_mask=value.valid_mask.clone(),
        branch=value.branch,
        positive_gt_ids=value.positive_gt_ids,
        unreachable_gt_ids=value.unreachable_gt_ids,
        reachable_gt_ids=value.reachable_gt_ids,
    )


@dataclass(frozen=True)
class InterventionAttempt:
    """One candidate and either its diagnostics/decision or a canonical error."""

    spec: TransformSpec
    diagnostics: TransformDiagnostics | None
    decision: AcceptanceDecision | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TransformSpec):
            raise TypeError("spec must be TransformSpec")
        if self.error is None:
            if not isinstance(self.diagnostics, TransformDiagnostics):
                raise TypeError("a completed attempt requires diagnostics")
            if not isinstance(self.decision, AcceptanceDecision):
                raise TypeError("a completed attempt requires an acceptance decision")
        else:
            if self.error not in _TRANSFORM_ERRORS:
                raise ValueError("unknown transform-attempt error code")
            if self.diagnostics is not None or self.decision is not None:
                raise ValueError("a failed transform attempt cannot carry a decision")

    @property
    def accepted(self) -> bool:
        return self.error is None and bool(self.decision and self.decision.accepted)


@dataclass(frozen=True)
class LegalInterventionReceipt:
    """Reproducibility record for the selected minimal legal candidate."""

    sample_id: str
    target_gt_id: int
    transform: TransformSpec
    transform_config_fingerprint: str
    occupancy_config_fingerprint: str
    match_config_fingerprint: str
    acceptance_config_fingerprint: str
    source_image_fingerprint: str
    gt_labels_fingerprint: str
    target_mask_fingerprint: str
    protected_mask_fingerprint: str
    transformed_image_fingerprint: str
    probability_before_fingerprint: str
    probability_fingerprint: str
    feature_fingerprint: str
    base_fingerprint: str
    occupancy_threshold: float
    before_covered_gt_ids: tuple[int, ...]
    after_covered_gt_ids: tuple[int, ...]
    retained_lineage_ious: tuple[tuple[int, float], ...]
    full_gt_recoverable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if (
            isinstance(self.target_gt_id, bool)
            or not isinstance(self.target_gt_id, int)
            or self.target_gt_id < 1
        ):
            raise ValueError("target_gt_id must be a positive integer")
        if not isinstance(self.transform, TransformSpec):
            raise TypeError("transform must be TransformSpec")
        for name in (
            "transform_config_fingerprint",
            "occupancy_config_fingerprint",
            "match_config_fingerprint",
            "acceptance_config_fingerprint",
            "source_image_fingerprint",
            "gt_labels_fingerprint",
            "target_mask_fingerprint",
            "protected_mask_fingerprint",
            "transformed_image_fingerprint",
            "probability_before_fingerprint",
            "probability_fingerprint",
            "feature_fingerprint",
            "base_fingerprint",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if _config_fingerprint(self.transform.config) != (
            self.transform_config_fingerprint
        ):
            raise ValueError("transform config does not match its fingerprint")
        if not 0.0 <= float(self.occupancy_threshold) <= 1.0:
            raise ValueError("occupancy_threshold must lie in [0,1]")
        if _config_fingerprint(
            OccupancyConfig(threshold=float(self.occupancy_threshold))
        ) != self.occupancy_config_fingerprint:
            raise ValueError("occupancy threshold does not match its fingerprint")
        if self.before_covered_gt_ids != tuple(
            sorted(set(self.before_covered_gt_ids))
        ):
            raise ValueError("before_covered_gt_ids must be sorted and unique")
        if self.after_covered_gt_ids != tuple(
            sorted(set(self.after_covered_gt_ids))
        ):
            raise ValueError("after_covered_gt_ids must be sorted and unique")
        if set(self.after_covered_gt_ids) != (
            set(self.before_covered_gt_ids) - {self.target_gt_id}
        ):
            raise ValueError("receipt does not describe one atomic coverage loss")
        if not self.full_gt_recoverable:
            raise ValueError("a legal receipt must be full-GT recoverable")


@dataclass(frozen=True, eq=False)
class ModelConsistentState:
    """Training state whose image, probability, feature and occupancy co-occur."""

    source_image: Tensor
    transformed_image: Tensor
    probability_before: Tensor
    probability: Tensor
    feature: Tensor
    occupancy: Tensor
    gt: InstanceMap
    prediction_before: InstanceMap
    matching_before: MatchResult
    prediction: InstanceMap
    matching: MatchResult
    occupancy_config: OccupancyConfig
    match_config: MatchConfig
    acceptance_config: AcceptanceConfig
    transform_diagnostics: TransformDiagnostics
    acceptance_decision: AcceptanceDecision
    supervision: BranchSupervision
    receipt: LegalInterventionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, LegalInterventionReceipt):
            raise TypeError("receipt must be LegalInterventionReceipt")
        if not isinstance(self.source_image, Tensor) or not isinstance(
            self.transformed_image, Tensor
        ):
            raise TypeError("source_image and transformed_image must be tensors")
        if self.source_image.ndim != 4 or self.source_image.shape[0] != 1:
            raise ValueError("source_image must have shape [1,C,H,W]")
        if self.transformed_image.ndim != 4 or self.transformed_image.shape[0] != 1:
            raise ValueError("transformed_image must have shape [1,C,H,W]")
        if self.source_image.shape != self.transformed_image.shape:
            raise ValueError("source and transformed image shapes must agree")
        if self.source_image.device != self.transformed_image.device:
            raise ValueError("source and transformed images must share a device")
        if not isinstance(self.probability_before, Tensor) or not isinstance(
            self.probability, Tensor
        ) or not isinstance(self.feature, Tensor):
            raise TypeError("probability_before, probability and feature must be tensors")
        if self.probability_before.ndim != 4 or self.probability_before.shape[:2] != (1, 1):
            raise ValueError("probability_before must have shape [1,1,H,W]")
        if self.probability.ndim != 4 or self.probability.shape[:2] != (1, 1):
            raise ValueError("probability must have shape [1,1,H,W]")
        if self.feature.ndim != 4 or self.feature.shape[0] != 1:
            raise ValueError("feature must have shape [1,C,h,w]")
        if (
            self.probability_before.dtype != torch.float32
            or self.probability.dtype != torch.float32
        ):
            raise TypeError("base probabilities must be float32")
        if self.feature.dtype != torch.float32:
            raise TypeError("feature must be float32")
        if not (
            self.probability_before.device
            == self.probability.device
            == self.feature.device
        ):
            raise ValueError("base probabilities and feature must share a device")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (self.probability_before, self.probability, self.feature)
        ):
            raise ValueError("base probabilities and feature must be finite")
        if any(
            value.requires_grad
            for value in (self.probability_before, self.probability, self.feature)
        ):
            raise ValueError("model-consistent base outputs must be detached")
        if not isinstance(self.occupancy, Tensor):
            raise TypeError("occupancy must be a tensor")
        if self.occupancy.device.type != "cpu" or self.occupancy.dtype != torch.bool:
            raise TypeError("occupancy must be a CPU bool tensor")
        if not isinstance(self.gt, InstanceMap):
            raise TypeError("gt must be an InstanceMap")
        if not isinstance(self.prediction_before, InstanceMap):
            raise TypeError("prediction_before must be an InstanceMap")
        if not isinstance(self.matching_before, MatchResult):
            raise TypeError("matching_before must be MatchResult")
        if not isinstance(self.prediction, InstanceMap):
            raise TypeError("prediction must be an InstanceMap")
        if not isinstance(self.match_config, MatchConfig):
            raise TypeError("match_config must be MatchConfig")
        if not isinstance(self.occupancy_config, OccupancyConfig):
            raise TypeError("occupancy_config must be OccupancyConfig")
        if not isinstance(self.acceptance_config, AcceptanceConfig):
            raise TypeError("acceptance_config must be AcceptanceConfig")
        if not isinstance(self.transform_diagnostics, TransformDiagnostics):
            raise TypeError("transform_diagnostics must be TransformDiagnostics")
        if not isinstance(self.acceptance_decision, AcceptanceDecision):
            raise TypeError("acceptance_decision must be AcceptanceDecision")
        if not self.acceptance_decision.accepted:
            raise ValueError("model-consistent state requires an accepted decision")
        _, expected_prediction_before = build_occupancy(
            self.probability_before, self.occupancy_config
        )
        if not torch.equal(
            expected_prediction_before.labels, self.prediction_before.labels
        ):
            raise ValueError(
                "prediction_before was not built from probability_before"
            )
        if self.occupancy.ndim != 2 or tuple(self.occupancy.shape) != self.prediction.shape:
            raise ValueError("occupancy and prediction grids must agree")
        if not torch.equal(self.occupancy, self.prediction.occupancy):
            raise ValueError("occupancy is inconsistent with prediction")
        if not isinstance(self.matching, MatchResult):
            raise TypeError("matching must be MatchResult")
        if self.matching_before != match_components(
            self.prediction_before, self.gt, self.match_config
        ):
            raise ValueError("matching_before is stale or inconsistent")
        if self.matching != match_components(
            self.prediction, self.gt, self.match_config
        ):
            raise ValueError("matching is stale or inconsistent")
        if _config_fingerprint(self.match_config) != (
            self.receipt.match_config_fingerprint
        ):
            raise ValueError("match_config does not match its receipt")
        if _config_fingerprint(self.occupancy_config) != (
            self.receipt.occupancy_config_fingerprint
        ):
            raise ValueError("occupancy_config does not match its receipt")
        if _config_fingerprint(self.acceptance_config) != (
            self.receipt.acceptance_config_fingerprint
        ):
            raise ValueError("acceptance_config does not match its receipt")
        if self.occupancy_config.threshold != self.receipt.occupancy_threshold:
            raise ValueError("occupancy threshold does not match its receipt")
        if _tensor_fingerprint(self.gt.labels) != self.receipt.gt_labels_fingerprint:
            raise ValueError("GT instance map does not match its receipt")
        if _tensor_fingerprint(self.gt.by_id(self.receipt.target_gt_id).mask) != (
            self.receipt.target_mask_fingerprint
        ):
            raise ValueError("target GT does not match its receipt")
        protected = union_instance_masks(
            self.gt,
            tuple(
                gt_id for gt_id in self.gt.ids
                if gt_id != self.receipt.target_gt_id
            ),
        )
        if _tensor_fingerprint(protected) != self.receipt.protected_mask_fingerprint:
            raise ValueError("protected GT mask does not match its receipt")
        if tuple(sorted(self.matching_before.matched_gt_ids)) != (
            self.receipt.before_covered_gt_ids
        ):
            raise ValueError("pre-intervention coverage does not match its receipt")
        if tuple(sorted(self.matching.matched_gt_ids)) != (
            self.receipt.after_covered_gt_ids
        ):
            raise ValueError("post-intervention coverage does not match its receipt")
        if not isinstance(self.supervision, BranchSupervision):
            raise TypeError("supervision must be BranchSupervision")
        if not torch.equal(self.supervision.occupancy.cpu()[0], self.occupancy):
            raise ValueError("supervision occupancy is inconsistent with state")
        if self.supervision.branch != "synthetic":
            raise ValueError("model-consistent intervention uses the synthetic branch")
        if self.supervision.positive_gt_ids != (self.receipt.target_gt_id,):
            raise ValueError("supervision and receipt target IDs disagree")
        selected = self.gt.by_id(self.receipt.target_gt_id).mask
        background = ~union_instance_masks(self.gt, self.gt.ids)
        writable = ~self.occupancy
        expected_target = (selected & writable).to(torch.float32).unsqueeze(0)
        expected_valid = (writable & (background | selected)).unsqueeze(0)
        if not torch.equal(self.supervision.target.cpu(), expected_target):
            raise ValueError("supervision target is inconsistent with GT and state")
        if not torch.equal(self.supervision.valid_mask.cpu(), expected_valid):
            raise ValueError("supervision valid mask is inconsistent with GT and state")
        if _tensor_fingerprint(self.transformed_image) != (
            self.receipt.transformed_image_fingerprint
        ):
            raise ValueError("transformed image does not match its receipt")
        if _tensor_fingerprint(self.source_image) != self.receipt.source_image_fingerprint:
            raise ValueError("source image does not match its receipt")
        if _tensor_fingerprint(self.probability_before) != (
            self.receipt.probability_before_fingerprint
        ):
            raise ValueError("pre-intervention probability does not match its receipt")
        if _tensor_fingerprint(self.probability) != self.receipt.probability_fingerprint:
            raise ValueError("probability does not match its receipt")
        if _tensor_fingerprint(self.feature) != self.receipt.feature_fingerprint:
            raise ValueError("feature does not match its receipt")
        expected = (
            self.probability.detach().to(device="cpu")[0, 0]
            >= self.receipt.occupancy_threshold
        )
        if not torch.equal(expected, self.occupancy):
            raise ValueError("occupancy was not thresholded from state probability")

        replayed_image, replayed_diagnostics = apply_counterfactual_transform(
            self.source_image,
            selected,
            self.receipt.transform,
            protected_mask=protected,
        )
        if not torch.equal(replayed_image, self.transformed_image):
            raise ValueError("transformed image is not a replay of its transform")
        if replayed_diagnostics != self.transform_diagnostics:
            raise ValueError("transform diagnostics are stale or inconsistent")
        changed_support = torch.any(
            self.transformed_image != self.source_image, dim=(0, 1)
        ).to(device="cpu", dtype=torch.bool)
        replayed_decision = assess_legal_intervention(
            gt=self.gt,
            target_gt_id=self.receipt.target_gt_id,
            pred_before=self.prediction_before,
            match_before=self.matching_before,
            probability_before=self.probability_before,
            pred_after=self.prediction,
            match_after=self.matching,
            probability_after=self.probability,
            changed_support=changed_support,
            transform_max_abs_delta=self.transform_diagnostics.max_abs_delta,
            transform_mean_abs_delta=self.transform_diagnostics.mean_abs_delta,
            transform_outside_max_delta=(
                self.transform_diagnostics.outside_roi_max_delta
            ),
            occupancy_config=self.occupancy_config,
            match_config=self.match_config,
            config=self.acceptance_config,
        )
        if replayed_decision != self.acceptance_decision:
            raise ValueError("acceptance decision is stale or inconsistent")
        if self.acceptance_decision.retained_lineage_ious != (
            self.receipt.retained_lineage_ious
        ) or self.acceptance_decision.full_gt_recoverable != (
            self.receipt.full_gt_recoverable
        ):
            raise ValueError("acceptance decision does not match its receipt")

    def branch_batch(self):
        """Return the existing training-engine batch on the feature device."""

        from ..train.step import BranchBatch

        self.validate_integrity()
        device = self.feature.device
        return BranchBatch(
            feature=self.feature.detach().clone(),
            occupancy=self.supervision.occupancy.to(device=device).clone().unsqueeze(0),
            target=self.supervision.target.to(device=device).clone().unsqueeze(0),
            valid_mask=self.supervision.valid_mask.to(device=device).clone().unsqueeze(0),
        )

    def validate_integrity(self) -> None:
        """Replay all constructor checks before crossing a cache/train boundary."""

        self.__post_init__()


@dataclass(frozen=True, eq=False)
class CounterfactualSearchResult:
    """Search result that makes unsupported targets explicit rather than fake."""

    sample_id: str
    target_gt_id: int
    match_before: MatchResult
    attempts: tuple[InterventionAttempt, ...]
    state: ModelConsistentState | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if (
            isinstance(self.target_gt_id, bool)
            or not isinstance(self.target_gt_id, int)
            or self.target_gt_id < 1
        ):
            raise ValueError("target_gt_id must be a positive integer")
        if not isinstance(self.match_before, MatchResult):
            raise TypeError("match_before must be MatchResult")
        if not isinstance(self.attempts, tuple) or any(
            not isinstance(attempt, InterventionAttempt)
            for attempt in self.attempts
        ):
            raise TypeError("attempts must be a tuple of InterventionAttempt")
        strengths = tuple(attempt.spec.strength for attempt in self.attempts)
        if strengths != tuple(sorted(set(strengths))):
            raise ValueError("attempt strengths must be strictly increasing")
        if self.state is None and not self.failure_reason:
            raise ValueError("an empty result requires a failure_reason")
        if self.failure_reason is not None and self.failure_reason not in _SEARCH_FAILURES:
            raise ValueError("unknown search failure code")
        if self.state is not None and self.failure_reason is not None:
            raise ValueError("a successful result cannot carry a failure_reason")
        accepted = tuple(attempt for attempt in self.attempts if attempt.accepted)
        if self.state is not None and len(accepted) != 1:
            raise ValueError("a successful minimal search must stop at one acceptance")
        if self.state is None and accepted:
            raise ValueError("a failed search cannot contain an accepted attempt")
        if self.state is not None:
            if self.sample_id != self.state.receipt.sample_id:
                raise ValueError("result and state sample IDs disagree")
            if self.target_gt_id != self.state.receipt.target_gt_id:
                raise ValueError("result and state target IDs disagree")
            if self.match_before != self.state.matching_before:
                raise ValueError("result and state pre-intervention matching disagree")
            selected_attempt = accepted[0]
            if not self.attempts or not self.attempts[-1].accepted:
                raise ValueError("search must stop immediately after acceptance")
            if selected_attempt.spec != self.state.receipt.transform:
                raise ValueError("accepted attempt and receipt transforms disagree")
            if selected_attempt.diagnostics != self.state.transform_diagnostics:
                raise ValueError("accepted attempt and state diagnostics disagree")
            if selected_attempt.decision != self.state.acceptance_decision:
                raise ValueError("accepted attempt and state decisions disagree")


def search_minimal_legal_intervention(
    *,
    sample_id: str,
    image: Tensor,
    gt: InstanceMap,
    target_gt_id: int,
    base: FrozenBaseAdapter,
    transform_config: TransformConfig = TransformConfig(),
    occupancy_config: OccupancyConfig = OccupancyConfig(),
    match_config: MatchConfig = MatchConfig(),
    acceptance_config: AcceptanceConfig = AcceptanceConfig(),
) -> CounterfactualSearchResult:
    """Return the weakest legal transform in the frozen strength grid.

    Every candidate is passed through ``base`` exactly once.  The selected
    probability, feature, occupancy and matching therefore all come from the
    same transformed input and the same frozen detector evaluation.
    """

    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if not isinstance(image, Tensor) or image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape [1,C,H,W]")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("image must be finite and floating point")
    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be an InstanceMap")
    if tuple(image.shape[-2:]) != gt.shape:
        raise ValueError("image and GT grids must agree")
    if not isinstance(base, FrozenBaseAdapter):
        raise TypeError("base must implement FrozenBaseAdapter")
    if not isinstance(transform_config, TransformConfig):
        raise TypeError("transform_config must be TransformConfig")
    if not isinstance(occupancy_config, OccupancyConfig):
        raise TypeError("occupancy_config must be OccupancyConfig")
    if not isinstance(match_config, MatchConfig):
        raise TypeError("match_config must be MatchConfig")
    if not isinstance(acceptance_config, AcceptanceConfig):
        raise TypeError("acceptance_config must be AcceptanceConfig")
    if (
        isinstance(target_gt_id, bool)
        or not isinstance(target_gt_id, int)
        or target_gt_id < 1
    ):
        raise ValueError("target_gt_id must be a positive integer")
    target = gt.by_id(target_gt_id).mask
    protected = union_instance_masks(
        gt, tuple(gt_id for gt_id in gt.ids if gt_id != target_gt_id)
    )

    before_output = base(image)
    _, pred_before = build_occupancy(before_output.probability, occupancy_config)
    match_before = match_components(pred_before, gt, match_config)
    if target_gt_id not in match_before.matched_gt_ids:
        return CounterfactualSearchResult(
            sample_id=sample_id,
            target_gt_id=target_gt_id,
            match_before=match_before,
            attempts=(),
            state=None,
            failure_reason="target_not_matched_before",
        )

    attempts: list[InterventionAttempt] = []
    for strength in transform_config.strength_grid:
        spec = TransformSpec(config=transform_config, strength=strength)
        try:
            transformed, diagnostics = apply_counterfactual_transform(
                image,
                target,
                spec,
                protected_mask=protected,
            )
        except InvalidTransformSupportError:
            attempts.append(
                InterventionAttempt(
                    spec=spec,
                    diagnostics=None,
                    decision=None,
                    error="invalid_transform_support",
                )
            )
            continue

        spatial_delta = torch.any(transformed != image, dim=(0, 1)).to(
            device="cpu", dtype=torch.bool
        )
        if not bool(torch.any(spatial_delta)):
            attempts.append(
                InterventionAttempt(
                    spec=spec,
                    diagnostics=None,
                    decision=None,
                    error="transform_has_no_effect",
                )
            )
            continue

        after_output = base(transformed)
        occupancy_after, pred_after = build_occupancy(
            after_output.probability, occupancy_config
        )
        match_after = match_components(pred_after, gt, match_config)
        decision = assess_legal_intervention(
            gt=gt,
            target_gt_id=target_gt_id,
            pred_before=pred_before,
            match_before=match_before,
            probability_before=before_output.probability,
            pred_after=pred_after,
            match_after=match_after,
            probability_after=after_output.probability,
            changed_support=spatial_delta,
            transform_max_abs_delta=diagnostics.max_abs_delta,
            transform_mean_abs_delta=diagnostics.mean_abs_delta,
            transform_outside_max_delta=diagnostics.outside_roi_max_delta,
            occupancy_config=occupancy_config,
            match_config=match_config,
            config=acceptance_config,
        )
        attempt = InterventionAttempt(
            spec=spec,
            diagnostics=diagnostics,
            decision=decision,
        )
        attempts.append(attempt)
        if not decision.accepted:
            continue

        supervision = build_atomic_intervention_supervision(
            occupancy_after,
            gt,
            target_gt_id,
            pred_before,
            match_before,
            match_after,
            match_config,
        )
        receipt = LegalInterventionReceipt(
            sample_id=sample_id,
            target_gt_id=target_gt_id,
            transform=spec,
            transform_config_fingerprint=_config_fingerprint(transform_config),
            occupancy_config_fingerprint=_config_fingerprint(occupancy_config),
            match_config_fingerprint=_config_fingerprint(match_config),
            acceptance_config_fingerprint=_config_fingerprint(acceptance_config),
            source_image_fingerprint=_tensor_fingerprint(image),
            gt_labels_fingerprint=_tensor_fingerprint(gt.labels),
            target_mask_fingerprint=_tensor_fingerprint(target),
            protected_mask_fingerprint=_tensor_fingerprint(protected),
            transformed_image_fingerprint=_tensor_fingerprint(transformed),
            probability_before_fingerprint=_tensor_fingerprint(
                before_output.probability
            ),
            probability_fingerprint=_tensor_fingerprint(after_output.probability),
            feature_fingerprint=_tensor_fingerprint(after_output.feature),
            base_fingerprint=base.fingerprint,
            occupancy_threshold=occupancy_config.threshold,
            before_covered_gt_ids=tuple(sorted(match_before.matched_gt_ids)),
            after_covered_gt_ids=tuple(sorted(match_after.matched_gt_ids)),
            retained_lineage_ious=decision.retained_lineage_ious,
            full_gt_recoverable=decision.full_gt_recoverable,
        )
        state = ModelConsistentState(
            source_image=image.detach().clone(),
            transformed_image=transformed.detach().clone(),
            probability_before=before_output.probability.detach().clone(),
            probability=after_output.probability.detach().clone(),
            feature=after_output.feature.detach().clone(),
            occupancy=occupancy_after.clone(),
            gt=_clone_instance_map(gt),
            prediction_before=_clone_instance_map(pred_before),
            matching_before=match_before,
            prediction=_clone_instance_map(pred_after),
            matching=match_after,
            occupancy_config=occupancy_config,
            match_config=match_config,
            acceptance_config=acceptance_config,
            transform_diagnostics=diagnostics,
            acceptance_decision=decision,
            supervision=_clone_supervision(supervision),
            receipt=receipt,
        )
        return CounterfactualSearchResult(
            sample_id=sample_id,
            target_gt_id=target_gt_id,
            match_before=match_before,
            attempts=tuple(attempts),
            state=state,
            failure_reason=None,
        )

    return CounterfactualSearchResult(
        sample_id=sample_id,
        target_gt_id=target_gt_id,
        match_before=match_before,
        attempts=tuple(attempts),
        state=None,
        failure_reason="no_legal_candidate",
    )


__all__ = [
    "CounterfactualSearchResult",
    "InterventionAttempt",
    "LegalInterventionReceipt",
    "ModelConsistentState",
    "search_minimal_legal_intervention",
]
