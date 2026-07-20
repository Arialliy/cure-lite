"""Value objects for predictor-induced supervision uncensoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from math import isfinite

import torch
from torch import Tensor

from ..types import LegalDeletion
from .protocol import CUREProtocol


DESCRIPTOR_FIELDS = (
    "target_score_min",
    "target_score_mean",
    "target_score_max",
    "local_scr",
    "log_area",
    "background_score_max",
    "normalized_boundary_distance",
)

_ELIGIBLE_CATALOG_SEAL = object()
_OOF_PROPENSITY_SEAL = object()
_INTERVENTION_CATALOG_SEAL = object()


def _positive_id(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, eq=False)
class TargetDescriptor:
    """A source-domain GT descriptor available for misses and detections.

    ``role`` is the state-selection indicator and is deliberately stored
    outside ``values``.  Including it, a match ID, or another outcome-only
    feature in the descriptor would make density-ratio estimation tautological.
    """

    sample_id: str
    group_id: str
    gt_id: int
    role: str
    values: Tensor
    split_role: str = "D_R"

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a non-empty string")
        _positive_id("gt_id", self.gt_id)
        if self.role not in {"factual_miss", "legal_covered"}:
            raise ValueError("role must be factual_miss or legal_covered")
        if self.split_role != "D_R":
            raise ValueError("propensity descriptors are source-only D_R artifacts")
        if not isinstance(self.values, Tensor) or self.values.ndim != 1:
            raise ValueError("values must be a one-dimensional tensor")
        if self.values.shape[0] != len(DESCRIPTOR_FIELDS):
            raise ValueError(
                f"descriptor must have {len(DESCRIPTOR_FIELDS)} fields"
            )
        if self.values.device.type != "cpu" or self.values.dtype != torch.float64:
            raise TypeError("descriptor values must be a CPU float64 tensor")
        if self.values.requires_grad:
            raise ValueError("descriptor values must be detached")
        if not torch.isfinite(self.values).all():
            raise ValueError("descriptor contains non-finite values")

    @property
    def missed(self) -> bool:
        return self.role == "factual_miss"

    @property
    def covered(self) -> bool:
        return self.role == "legal_covered"

    @property
    def key(self) -> tuple[str, int]:
        return self.sample_id, self.gt_id


@dataclass(frozen=True)
class EligibleSampleCatalog:
    """One sample's canonical eligible universe and legal support."""

    sample_id: str
    group_id: str
    base_fingerprint: str
    protocol_fingerprint: str
    frozen_output_fingerprint: str
    source_fingerprint: str
    gt_fingerprint: str
    descriptors: tuple[TargetDescriptor, ...]
    legal_deletions: tuple[LegalDeletion, ...]
    excluded_factual_gt_ids: tuple[int, ...]
    excluded_covered_gt_ids: tuple[int, ...]
    occupancy_threshold: float
    suppression_radius: int
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        issuing = self._seal is _ELIGIBLE_CATALOG_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _ELIGIBLE_CATALOG_SEAL
        ):
            raise ValueError("eligible catalog was not issued by the canonical builder")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be non-empty")
        if not isinstance(self.base_fingerprint, str) or not self.base_fingerprint:
            raise ValueError("base_fingerprint must be non-empty")
        for name in (
            "protocol_fingerprint",
            "frozen_output_fingerprint",
            "source_fingerprint",
            "gt_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if isinstance(self.suppression_radius, bool) or not isinstance(
            self.suppression_radius, int
        ) or self.suppression_radius < 0:
            raise ValueError("suppression_radius must be a non-negative integer")
        if not isfinite(float(self.occupancy_threshold)) or not 0.0 <= float(
            self.occupancy_threshold
        ) <= 1.0:
            raise ValueError("occupancy_threshold must lie in [0,1]")
        if any(not isinstance(item, TargetDescriptor) for item in self.descriptors):
            raise TypeError("descriptors must contain TargetDescriptor")
        descriptor_keys = tuple(item.key for item in self.descriptors)
        if descriptor_keys != tuple(sorted(descriptor_keys)) or len(
            set(descriptor_keys)
        ) != len(descriptor_keys):
            raise ValueError("descriptors must have unique sorted keys")
        if any(
            item.sample_id != self.sample_id or item.group_id != self.group_id
            for item in self.descriptors
        ):
            raise ValueError("descriptor identity differs from the sample catalog")
        if any(not isinstance(item, LegalDeletion) for item in self.legal_deletions):
            raise TypeError("legal_deletions must contain LegalDeletion")
        deletion_ids = tuple(item.gt_id for item in self.legal_deletions)
        if deletion_ids != tuple(sorted(set(deletion_ids))):
            raise ValueError("legal deletions must have unique sorted target IDs")
        legal_descriptor_ids = tuple(
            item.gt_id for item in self.descriptors if item.covered
        )
        if legal_descriptor_ids != deletion_ids:
            raise ValueError("legal descriptor universe and deletion support differ")
        for name, ids in (
            ("excluded_factual_gt_ids", self.excluded_factual_gt_ids),
            ("excluded_covered_gt_ids", self.excluded_covered_gt_ids),
        ):
            if ids != tuple(sorted(set(ids))) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in ids
            ):
                raise ValueError(f"{name} must contain unique sorted positive IDs")
        eligible_ids = {item.gt_id for item in self.descriptors}
        if eligible_ids & (
            set(self.excluded_factual_gt_ids) | set(self.excluded_covered_gt_ids)
        ):
            raise ValueError("eligible and excluded target IDs must be disjoint")
        if issuing:
            object.__setattr__(
                self, "_seal", (_ELIGIBLE_CATALOG_SEAL, self.fingerprint)
            )
        elif self._seal[1] != self.fingerprint:
            raise ValueError("eligible catalog content differs from its receipt")

    @property
    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(
            repr(
                (
                    self.sample_id,
                    self.group_id,
                    self.base_fingerprint,
                    self.protocol_fingerprint,
                    self.frozen_output_fingerprint,
                    self.source_fingerprint,
                    self.gt_fingerprint,
                    self.excluded_factual_gt_ids,
                    self.excluded_covered_gt_ids,
                    self.occupancy_threshold,
                    self.suppression_radius,
                )
            ).encode("utf-8")
        )
        for item in self.descriptors:
            hasher.update(
                repr(
                    (
                        item.sample_id,
                        item.group_id,
                        item.gt_id,
                        item.role,
                        tuple(float(value) for value in item.values),
                    )
                ).encode("utf-8")
            )
        for deletion in self.legal_deletions:
            hasher.update(
                repr(
                    (
                        deletion.gt_id,
                        deletion.pred_id,
                        tuple(
                            (pair.gt_id, pair.pred_id)
                            for pair in deletion.match_after.pairs
                        ),
                    )
                ).encode("utf-8")
            )
            hasher.update(
                deletion.occupancy_after.contiguous().numpy().tobytes(order="C")
            )
        return hasher.hexdigest()

    def validate_receipt(self) -> None:
        if not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _ELIGIBLE_CATALOG_SEAL
            and self._seal[1] == self.fingerprint
        ):
            raise ValueError("eligible catalog content differs from its receipt")


@dataclass(frozen=True)
class PropensityEstimate:
    """One group-disjoint out-of-fold miss-state probability prediction."""

    sample_id: str
    group_id: str
    gt_id: int
    role: str
    fold: int
    raw_probability: float
    clipped_probability: float
    sampling_weight: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a non-empty string")
        _positive_id("gt_id", self.gt_id)
        if self.role not in {"factual_miss", "legal_covered"}:
            raise ValueError("role must be factual_miss or legal_covered")
        if isinstance(self.fold, bool) or not isinstance(self.fold, int) or self.fold < 0:
            raise ValueError("fold must be a non-negative integer")
        for name, value in (
            ("raw_probability", self.raw_probability),
            ("clipped_probability", self.clipped_probability),
        ):
            if not isfinite(float(value)) or not 0.0 < float(value) < 1.0:
                raise ValueError(f"{name} must lie strictly inside (0,1)")
        if self.covered:
            if self.sampling_weight is None:
                raise ValueError("covered targets require a sampling weight")
            if not isfinite(float(self.sampling_weight)) or self.sampling_weight <= 0.0:
                raise ValueError("sampling_weight must be finite and positive")
        elif self.sampling_weight is not None:
            raise ValueError("missed targets are not counterfactual candidates")

    @property
    def key(self) -> tuple[str, int]:
        return self.sample_id, self.gt_id

    @property
    def covered(self) -> bool:
        return self.role == "legal_covered"


@dataclass(frozen=True)
class OOFPropensityResult:
    """Complete out-of-fold estimates and overlap diagnostics."""

    estimates: tuple[PropensityEstimate, ...]
    fold_by_group: tuple[tuple[str, int], ...]
    descriptor_fingerprint: str
    manifest_fingerprint: str
    protocol_fingerprint: str
    effective_sample_size: float
    probability_clipped_fraction: float
    odds_capped_fraction: float
    brier_score: float
    factual_fraction: float
    factual_probability_range: tuple[float, float]
    legal_probability_range: tuple[float, float]
    overlap_interval: tuple[float, float] | None
    max_sampling_weight: float
    clip_epsilon: float
    max_odds: float
    smd_before: tuple[float, ...]
    smd_after: tuple[float, ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        issuing = self._seal is _OOF_PROPENSITY_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _OOF_PROPENSITY_SEAL
        ):
            raise ValueError("OOF result was not issued by canonical cross-fitting")
        if not self.estimates:
            raise ValueError("OOF propensity result cannot be empty")
        for name, value in (
            ("descriptor_fingerprint", self.descriptor_fingerprint),
            ("manifest_fingerprint", self.manifest_fingerprint),
            ("protocol_fingerprint", self.protocol_fingerprint),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        keys = tuple(item.key for item in self.estimates)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("propensity estimates must have unique sorted keys")
        group_items = tuple(group_id for group_id, _ in self.fold_by_group)
        if group_items != tuple(sorted(group_items)) or len(set(group_items)) != len(group_items):
            raise ValueError("fold_by_group must contain unique sorted group IDs")
        if not isfinite(float(self.effective_sample_size)) or self.effective_sample_size <= 0.0:
            raise ValueError("effective_sample_size must be finite and positive")
        for name, value in (
            ("probability_clipped_fraction", self.probability_clipped_fraction),
            ("odds_capped_fraction", self.odds_capped_fraction),
            ("brier_score", self.brier_score),
            ("factual_fraction", self.factual_fraction),
        ):
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        for name, bounds in (
            ("factual_probability_range", self.factual_probability_range),
            ("legal_probability_range", self.legal_probability_range),
        ):
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(not isfinite(float(value)) for value in bounds)
                or not 0.0 < float(bounds[0]) <= float(bounds[1]) < 1.0
            ):
                raise ValueError(f"{name} must be an ordered interval inside (0,1)")
        if self.overlap_interval is not None:
            bounds = self.overlap_interval
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(not isfinite(float(value)) for value in bounds)
                or not 0.0 < float(bounds[0]) <= float(bounds[1]) < 1.0
            ):
                raise ValueError("overlap_interval must be inside (0,1) or None")
        if not isfinite(float(self.max_sampling_weight)) or self.max_sampling_weight <= 0.0:
            raise ValueError("max_sampling_weight must be finite and positive")
        if not isfinite(float(self.clip_epsilon)) or not 0.0 < self.clip_epsilon < 0.5:
            raise ValueError("clip_epsilon must lie in (0,0.5)")
        if not isfinite(float(self.max_odds)) or self.max_odds <= 0.0:
            raise ValueError("max_odds must be finite and positive")
        if self.max_sampling_weight > self.max_odds:
            raise ValueError("max_sampling_weight may not exceed max_odds")
        if len(self.smd_before) != len(DESCRIPTOR_FIELDS) or len(self.smd_after) != len(
            DESCRIPTOR_FIELDS
        ):
            raise ValueError("SMD diagnostics must follow the descriptor schema")
        if any(not isfinite(float(value)) for value in (*self.smd_before, *self.smd_after)):
            raise ValueError("SMD diagnostics must be finite")
        if issuing:
            object.__setattr__(self, "_seal", (_OOF_PROPENSITY_SEAL, self.fingerprint))
        elif self._seal[1] != self.fingerprint:
            raise ValueError("OOF result content differs from its receipt")

    def by_key(self) -> dict[tuple[str, int], PropensityEstimate]:
        return {item.key: item for item in self.estimates}

    @property
    def fingerprint(self) -> str:
        payload = repr(
            (
                self.estimates,
                self.fold_by_group,
                self.descriptor_fingerprint,
                self.manifest_fingerprint,
                self.protocol_fingerprint,
                self.effective_sample_size,
                self.probability_clipped_fraction,
                self.odds_capped_fraction,
                self.brier_score,
                self.factual_fraction,
                self.factual_probability_range,
                self.legal_probability_range,
                self.overlap_interval,
                self.max_sampling_weight,
                self.clip_epsilon,
                self.max_odds,
                self.smd_before,
                self.smd_after,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_receipt(self) -> None:
        if not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _OOF_PROPENSITY_SEAL
            and self._seal[1] == self.fingerprint
        ):
            raise ValueError("OOF result content differs from its receipt")


@dataclass(frozen=True)
class WeightedCounterfactualCandidate:
    """One legal coverage deletion bound to an OOF-derived sampling weight."""

    sample_id: str
    deletion: LegalDeletion
    miss_probability: float
    weight: float
    max_odds: float

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.deletion, LegalDeletion):
            raise TypeError("deletion must be LegalDeletion")
        if not isfinite(float(self.miss_probability)) or not 0.0 < self.miss_probability < 1.0:
            raise ValueError("miss_probability must lie strictly inside (0,1)")
        if not isfinite(float(self.weight)) or self.weight <= 0.0:
            raise ValueError("weight must be finite and positive")
        if not isfinite(float(self.max_odds)) or self.max_odds <= 0.0:
            raise ValueError("max_odds must be finite and positive")
        expected = min(
            self.miss_probability / (1.0 - self.miss_probability),
            self.max_odds,
        )
        if not math.isclose(self.weight, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("weight must equal capped odds of miss_probability")

    @property
    def key(self) -> tuple[str, int, int]:
        return self.sample_id, self.deletion.gt_id, self.deletion.pred_id


@dataclass(frozen=True)
class CUREInterventionCatalog:
    """Immutable receipt binding the eligible universe to its odds support."""

    candidates: tuple[WeightedCounterfactualCandidate, ...]
    eligible_keys: tuple[tuple[str, int, str], ...]
    protocol: CUREProtocol
    frozen_output_fingerprints: tuple[tuple[str, str], ...]
    source_fingerprints: tuple[tuple[str, str], ...]
    gt_fingerprints: tuple[tuple[str, str], ...]
    propensity_fingerprint: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        issuing = self._seal is _INTERVENTION_CATALOG_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _INTERVENTION_CATALOG_SEAL
        ):
            raise ValueError("catalog is not bound to a canonical OOF receipt")
        if not self.candidates or any(
            not isinstance(item, WeightedCounterfactualCandidate)
            for item in self.candidates
        ):
            raise TypeError("candidates must be a non-empty weighted tuple")
        if not isinstance(self.protocol, CUREProtocol):
            raise TypeError("protocol must be CUREProtocol")
        self.protocol.validate_receipt()
        candidate_keys = tuple(item.key for item in self.candidates)
        if candidate_keys != tuple(sorted(candidate_keys)) or len(
            set(candidate_keys)
        ) != len(candidate_keys):
            raise ValueError("candidate keys must be unique and sorted")
        if not self.eligible_keys or self.eligible_keys != tuple(
            sorted(set(self.eligible_keys))
        ):
            raise ValueError("eligible_keys must be non-empty, unique, and sorted")
        for sample_id, gt_id, role in self.eligible_keys:
            if not sample_id or isinstance(gt_id, bool) or not isinstance(gt_id, int) or gt_id < 1:
                raise ValueError("eligible keys contain an invalid identity")
            if role not in {"factual_miss", "legal_covered"}:
                raise ValueError("eligible keys contain an invalid role")
        legal_keys = {
            (sample_id, gt_id)
            for sample_id, gt_id, role in self.eligible_keys
            if role == "legal_covered"
        }
        candidate_target_keys = {
            (item.sample_id, item.deletion.gt_id) for item in self.candidates
        }
        if candidate_target_keys != legal_keys:
            raise ValueError("candidate support differs from legal eligible keys")
        if not any(role == "factual_miss" for _, _, role in self.eligible_keys):
            raise ValueError("eligible universe must contain factual misses")
        eligible_sample_ids = {sample_id for sample_id, _, _ in self.eligible_keys}
        for name, rows in (
            ("frozen_output_fingerprints", self.frozen_output_fingerprints),
            ("source_fingerprints", self.source_fingerprints),
            ("gt_fingerprints", self.gt_fingerprints),
        ):
            if not rows or rows != tuple(sorted(set(rows))):
                raise ValueError(f"{name} must be non-empty, unique, and sorted")
            sample_ids = {sample_id for sample_id, _ in rows}
            if not eligible_sample_ids.issubset(sample_ids):
                raise ValueError(f"{name} does not cover the eligible universe")
            for sample_id, digest in rows:
                self.protocol.assert_sample(sample_id, split="D_R")
                if len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise ValueError(f"{name} contains an invalid SHA-256 digest")
        if (
            not isinstance(self.propensity_fingerprint, str)
            or len(self.propensity_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.propensity_fingerprint
            )
        ):
            raise ValueError("propensity_fingerprint must be a SHA-256 digest")
        if issuing:
            object.__setattr__(
                self, "_seal", (_INTERVENTION_CATALOG_SEAL, self.fingerprint)
            )
        elif self._seal[1] != self.fingerprint:
            raise ValueError("intervention catalog content differs from its receipt")

    @property
    def occupancy_threshold(self) -> float:
        return self.protocol.residual_config.occupancy_threshold

    @property
    def suppression_radius(self) -> int:
        return self.protocol.residual_config.suppression_radius

    @property
    def base_fingerprint(self) -> str:
        return self.protocol.base_fingerprint

    @property
    def manifest_fingerprint(self) -> str:
        return self.protocol.manifest_fingerprint

    @property
    def protocol_fingerprint(self) -> str:
        return self.protocol.fingerprint

    @property
    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.protocol.fingerprint.encode("ascii"))
        hasher.update(repr(self.eligible_keys).encode("utf-8"))
        hasher.update(repr(self.frozen_output_fingerprints).encode("utf-8"))
        hasher.update(repr(self.source_fingerprints).encode("utf-8"))
        hasher.update(repr(self.gt_fingerprints).encode("utf-8"))
        hasher.update(self.propensity_fingerprint.encode("ascii"))
        for item in self.candidates:
            hasher.update(
                repr(
                    (
                        item.key,
                        item.miss_probability,
                        item.weight,
                        item.max_odds,
                    )
                ).encode("utf-8")
            )
            occupancy = item.deletion.occupancy_after.contiguous().numpy()
            hasher.update(occupancy.tobytes(order="C"))
        return hasher.hexdigest()

    def validate_receipt(self) -> None:
        if not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _INTERVENTION_CATALOG_SEAL
            and self._seal[1] == self.fingerprint
        ):
            raise ValueError("intervention catalog content differs from its receipt")


@dataclass(frozen=True, eq=False)
class ResidualSetSupervision:
    """The explicitly supervised and ignored GTs for one residual state."""

    occupancy: Tensor
    editable_mask: Tensor
    target: Tensor
    background_mask: Tensor
    object_masks: Tensor
    positive_gt_ids: tuple[int, ...]
    uneditable_gt_ids: tuple[int, ...]
    ignored_gt_ids: tuple[int, ...]
    branch: str

    def __post_init__(self) -> None:
        fields = (self.occupancy, self.editable_mask, self.target, self.background_mask)
        if any(not isinstance(value, Tensor) for value in fields):
            raise TypeError("supervision masks must be tensors")
        if any(value.ndim != 3 or value.shape[0] != 1 for value in fields):
            raise ValueError("supervision masks must have shape [1,H,W]")
        if len({tuple(value.shape) for value in fields}) != 1:
            raise ValueError("supervision masks must share a grid")
        if self.occupancy.dtype != torch.bool or self.editable_mask.dtype != torch.bool:
            raise TypeError("occupancy and editable_mask must be bool")
        if self.background_mask.dtype != torch.bool or self.target.dtype != torch.float32:
            raise TypeError("background_mask must be bool and target float32")
        if len({value.device for value in fields}) != 1:
            raise ValueError("supervision masks must share a device")
        if not torch.isfinite(self.target).all() or torch.any(
            (self.target != 0.0) & (self.target != 1.0)
        ):
            raise ValueError("target must be finite and binary")
        if not isinstance(self.object_masks, Tensor) or self.object_masks.ndim != 3:
            raise ValueError("object_masks must have shape [K,H,W]")
        if self.object_masks.dtype != torch.bool or self.object_masks.device != self.target.device:
            raise TypeError("object_masks must be bool on the supervision device")
        if tuple(self.object_masks.shape[1:]) != tuple(self.target.shape[-2:]):
            raise ValueError("object_masks and supervision grid differ")
        if self.positive_gt_ids != tuple(sorted(set(self.positive_gt_ids))):
            raise ValueError("positive_gt_ids must be sorted and unique")
        if self.uneditable_gt_ids != tuple(sorted(set(self.uneditable_gt_ids))):
            raise ValueError("uneditable_gt_ids must be sorted and unique")
        if self.ignored_gt_ids != tuple(sorted(set(self.ignored_gt_ids))):
            raise ValueError("ignored_gt_ids must be sorted and unique")
        if len(self.positive_gt_ids) != self.object_masks.shape[0]:
            raise ValueError("one object mask is required per positive GT")
        for name, values in (
            ("positive_gt_ids", self.positive_gt_ids),
            ("uneditable_gt_ids", self.uneditable_gt_ids),
            ("ignored_gt_ids", self.ignored_gt_ids),
        ):
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
                raise ValueError(f"{name} must contain positive integers")
        id_sets = (
            set(self.positive_gt_ids),
            set(self.uneditable_gt_ids),
            set(self.ignored_gt_ids),
        )
        if any(id_sets[left] & id_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("positive, uneditable, and ignored GT IDs must be disjoint")
        if self.branch not in {"factual", "counterfactual"}:
            raise ValueError("branch must be factual or counterfactual")
        if torch.any(self.editable_mask & self.occupancy):
            raise ValueError("editable_mask may not overlap occupancy")
        if torch.any(self.background_mask & ~self.editable_mask):
            raise ValueError("background_mask must be editable")
        if torch.any(self.target.to(torch.bool) & ~self.editable_mask):
            raise ValueError("target must be editable")
        if self.object_masks.shape[0]:
            union = self.object_masks.any(dim=0, keepdim=True)
        else:
            union = torch.zeros_like(self.occupancy)
        if not torch.equal(union, self.target.to(torch.bool)):
            raise ValueError("target must equal the union of object_masks")
        if torch.any(self.background_mask & self.target.to(torch.bool)):
            raise ValueError("background and target masks must be disjoint")


__all__ = [
    "CUREInterventionCatalog",
    "DESCRIPTOR_FIELDS",
    "EligibleSampleCatalog",
    "OOFPropensityResult",
    "PropensityEstimate",
    "ResidualSetSupervision",
]
