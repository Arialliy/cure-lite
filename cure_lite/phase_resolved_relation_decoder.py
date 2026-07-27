"""CURE-Lite decoder with phase-resolved relation-controlled release.

The decoder is one coherent state equation, not a cascade of correction
modules.  A single shared projection supplies phase queries and coverage
keys.  Query direction participates in feature--coverage relevance, while
query norm supplies phase evidence.  Relevant Base occupancy suppresses that
evidence; unrelated occupancy does not.

External inference inputs remain exactly ``(F_b, O)``:

``F_b``
    detached feature map from an arbitrary frozen IRSTD detector;

``O``
    binary Base occupancy at output resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import expm1, log

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .phase_resolved_feature_coverage_relation import (
    PFCR_BURDEN_POLICY,
    PFCR_FEATURE_NORMALIZATION_POLICY,
    PFCR_OCCUPANCY_POLICY,
    PFCR_SMOOTH_RELATION_POLICY,
    PhaseResolvedFeatureCoverageRelation,
    PhaseResolvedFeatureCoverageRelationConfig,
    PhaseResolvedFeatureCoverageRelationFields,
)


PFCR_INITIAL_BASELINE_PROBABILITY = 0.01
PFCR_EVIDENCE_POLICY = (
    "dimension_normalized_bounded_query_energy_with_fixed_ceiling_v3"
)
PFCR_RELEASE_POLICY = (
    "evidence_times_one_minus_relevant_coverage_burden_v1"
)
PFCR_OUTPUT_POLICY = (
    "native_pixelshuffle_without_interpolation_then_hard_union_v1"
)
PFCR_EVIDENCE_CEILING = 10.0


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse softplus requires a positive value")
    return value + log(-expm1(-value))


@dataclass(frozen=True)
class PhaseResolvedRelationDecoderConfig:
    """Frozen CURE-Lite relation-controlled decoder contract."""

    feature_channels: int
    feature_stride: int
    relation_dim: int = 8
    normalization_epsilon: float = 1.0e-6
    baseline_probability: float = PFCR_INITIAL_BASELINE_PROBABILITY
    evidence_ceiling: float = PFCR_EVIDENCE_CEILING
    relation_policy: str = PFCR_SMOOTH_RELATION_POLICY
    feature_normalization_policy: str = (
        PFCR_FEATURE_NORMALIZATION_POLICY
    )
    occupancy_policy: str = PFCR_OCCUPANCY_POLICY
    burden_policy: str = PFCR_BURDEN_POLICY
    evidence_policy: str = PFCR_EVIDENCE_POLICY
    release_policy: str = PFCR_RELEASE_POLICY
    output_policy: str = PFCR_OUTPUT_POLICY

    def __post_init__(self) -> None:
        relation = self.to_relation_config()
        object.__setattr__(
            self,
            "feature_channels",
            relation.feature_channels,
        )
        object.__setattr__(
            self,
            "feature_stride",
            relation.feature_stride,
        )
        object.__setattr__(
            self,
            "relation_dim",
            relation.relation_dim,
        )
        object.__setattr__(
            self,
            "normalization_epsilon",
            relation.normalization_epsilon,
        )
        if (
            isinstance(self.baseline_probability, bool)
            or not isinstance(self.baseline_probability, float)
            or self.baseline_probability
            != PFCR_INITIAL_BASELINE_PROBABILITY
        ):
            raise ValueError(
                "CURE-Lite relation decoder fixes baseline_probability"
            )
        if (
            isinstance(self.evidence_ceiling, bool)
            or not isinstance(self.evidence_ceiling, float)
            or self.evidence_ceiling != PFCR_EVIDENCE_CEILING
        ):
            raise ValueError(
                "CURE-Lite relation decoder fixes evidence_ceiling"
            )
        frozen = {
            "relation_policy": PFCR_SMOOTH_RELATION_POLICY,
            "feature_normalization_policy": (
                PFCR_FEATURE_NORMALIZATION_POLICY
            ),
            "occupancy_policy": PFCR_OCCUPANCY_POLICY,
            "burden_policy": PFCR_BURDEN_POLICY,
            "evidence_policy": PFCR_EVIDENCE_POLICY,
            "release_policy": PFCR_RELEASE_POLICY,
            "output_policy": PFCR_OUTPUT_POLICY,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"CURE-Lite relation decoder fixes {name}"
                )

    @property
    def phase_channels(self) -> int:
        return self.feature_stride**2

    @property
    def expected_parameter_count(self) -> int:
        relation = self.to_relation_config()
        return relation.expected_parameter_count + 1

    def to_relation_config(
        self,
    ) -> PhaseResolvedFeatureCoverageRelationConfig:
        return PhaseResolvedFeatureCoverageRelationConfig(
            feature_channels=self.feature_channels,
            feature_stride=self.feature_stride,
            relation_dim=self.relation_dim,
            normalization_epsilon=self.normalization_epsilon,
            relation_policy=self.relation_policy,
            feature_normalization_policy=(
                self.feature_normalization_policy
            ),
            occupancy_policy=self.occupancy_policy,
            burden_policy=self.burden_policy,
        )


@dataclass(frozen=True)
class PhaseResolvedRelationDecoderFields:
    """Every field of one CURE-Lite relation decoder forward."""

    relation: PhaseResolvedFeatureCoverageRelationFields
    phase_evidence: Tensor
    release_gate: Tensor
    native_phase_evidence: Tensor
    native_baseline_logits: Tensor
    native_phase_logits: Tensor
    logits: Tensor
    completion_probability: Tensor
    output_size: tuple[int, int]


class CURELitePhaseResolvedRelationDecoder(nn.Module):
    """Detector-independent CURE-Lite relation-controlled decoder."""

    config: PhaseResolvedRelationDecoderConfig

    def __init__(
        self,
        config: PhaseResolvedRelationDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(config, PhaseResolvedRelationDecoderConfig):
            if feature_channels is not None or feature_stride is not None:
                raise ValueError(
                    "do not override an explicit decoder config"
                )
            resolved = config
        elif config is None:
            if feature_channels is None or feature_stride is None:
                raise TypeError(
                    "decoder config or feature_channels/feature_stride "
                    "is required"
                )
            resolved = PhaseResolvedRelationDecoderConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
            )
        else:
            raise TypeError(
                "config must be PhaseResolvedRelationDecoderConfig or None"
            )
        self.config = resolved
        self.relation = PhaseResolvedFeatureCoverageRelation(
            resolved.to_relation_config()
        )
        initial_logit = log(
            resolved.baseline_probability
            / (1.0 - resolved.baseline_probability)
        )
        self.baseline_raw = nn.Parameter(
            torch.tensor(
                _inverse_softplus(-initial_logit),
                dtype=torch.float32,
            )
        )
        self.pixel_shuffle = nn.PixelShuffle(resolved.feature_stride)
        actual = sum(
            parameter.numel() for parameter in self.parameters()
        )
        if actual != resolved.expected_parameter_count:
            raise AssertionError(
                "relation decoder parameter count differs from its contract"
            )

    @property
    def feature_channels(self) -> int:
        """Expose the generic frozen-feature input width."""

        return self.config.feature_channels

    @property
    def feature_stride(self) -> int:
        """Expose the exact native output stride without naming a detector."""

        return self.config.feature_stride

    def _validate_native_output_contract(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> tuple[int, int]:
        expected = (
            int(feature.shape[-2]) * self.config.feature_stride,
            int(feature.shape[-1]) * self.config.feature_stride,
        )
        actual = tuple(int(value) for value in occupancy.shape[-2:])
        if actual != expected:
            raise ValueError(
                "occupancy output size must equal feature size times "
                "feature_stride; relation decoder does not interpolate "
                "tiny-target logits"
            )
        return actual

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> PhaseResolvedRelationDecoderFields:
        """Return the full CURE-Lite state equation."""

        return self._forward_fields(
            feature,
            occupancy,
            audit=True,
        )

    def _forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
        *,
        audit: bool,
    ) -> PhaseResolvedRelationDecoderFields:
        """Execute the same state equation with optional repeated scans."""

        if not isinstance(audit, bool):
            raise TypeError("audit must be bool")
        relation = self.relation._forward_fields(
            feature,
            occupancy,
            audit=audit,
        )
        output_size = self._validate_native_output_contract(
            feature,
            occupancy,
        )
        squared_query_norm = relation.phase_evidence_strength.square()
        phase_evidence = (
            self.config.evidence_ceiling
            * squared_query_norm
            / (
                float(self.config.relation_dim)
                + squared_query_norm
            )
        )
        release_gate = 1.0 - relation.coverage_burden
        native_evidence = phase_evidence * release_gate
        baseline_value = -F.softplus(self.baseline_raw)
        native_baseline = torch.ones_like(native_evidence) * baseline_value
        native_logits = native_baseline + native_evidence
        logits = self.pixel_shuffle(native_logits)
        completion_probability = torch.sigmoid(logits)

        fields = PhaseResolvedRelationDecoderFields(
            relation=relation,
            phase_evidence=phase_evidence.contiguous(),
            release_gate=release_gate.contiguous(),
            native_phase_evidence=native_evidence.contiguous(),
            native_baseline_logits=native_baseline.contiguous(),
            native_phase_logits=native_logits.contiguous(),
            logits=logits.contiguous(),
            completion_probability=completion_probability.contiguous(),
            output_size=output_size,
        )
        if audit:
            self._validate_fields(
                fields,
                feature=feature,
                occupancy=occupancy,
            )
        return fields

    def _validate_fields(
        self,
        fields: PhaseResolvedRelationDecoderFields,
        *,
        feature: Tensor,
        occupancy: Tensor,
    ) -> None:
        native_shape = (
            int(feature.shape[0]),
            self.config.phase_channels,
            int(feature.shape[-2]),
            int(feature.shape[-1]),
        )
        output_shape = tuple(int(value) for value in occupancy.shape)
        for name, value in (
            ("phase_evidence", fields.phase_evidence),
            ("release_gate", fields.release_gate),
            ("native_phase_evidence", fields.native_phase_evidence),
            (
                "native_baseline_logits",
                fields.native_baseline_logits,
            ),
            ("native_phase_logits", fields.native_phase_logits),
        ):
            if tuple(value.shape) != native_shape:
                raise AssertionError(f"{name} has an invalid shape")
            if value.dtype != feature.dtype or value.device != feature.device:
                raise ValueError(
                    f"{name} must match feature dtype/device"
                )
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"{name} must be finite")
        for name, value in (
            ("logits", fields.logits),
            (
                "completion_probability",
                fields.completion_probability,
            ),
        ):
            if tuple(value.shape) != output_shape:
                raise AssertionError(f"{name} has an invalid shape")
            if value.dtype != feature.dtype or value.device != feature.device:
                raise ValueError(
                    f"{name} must match feature dtype/device"
                )
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"{name} must be finite")
        bounds = torch.stack(
            (
                (fields.phase_evidence >= 0.0).all(),
                (fields.release_gate >= 0.0).all(),
                (fields.release_gate <= 1.0).all(),
                (fields.native_phase_evidence >= 0.0).all(),
                (fields.native_baseline_logits < 0.0).all(),
                (fields.completion_probability >= 0.0).all(),
                (fields.completion_probability <= 1.0).all(),
            )
        )
        if not bool(bounds.all()):
            raise AssertionError("relation decoder field bounds changed")

    def forward(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        """Return high-resolution completion logits."""

        return self.forward_fields(feature, occupancy).logits

    def forward_training_logits(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        """Return identical logits without repeated full-field scans.

        This path is reserved for an already-validated immutable cache inside
        the formal optimizer loop.  Public inference remains fully audited.
        """

        return self._forward_fields(
            feature,
            occupancy,
            audit=False,
        ).logits

    def predict_completion(
        self,
        feature: Tensor,
        occupancy: Tensor,
        *,
        threshold: float,
    ) -> Tensor:
        """Threshold completion probability with a strict frozen rule."""

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, float)
            or not 0.0 < threshold < 1.0
        ):
            raise ValueError("threshold must be a float in (0,1)")
        probability = self.forward_fields(
            feature,
            occupancy,
        ).completion_probability
        return (probability > threshold).contiguous()

    def predict_union(
        self,
        feature: Tensor,
        occupancy: Tensor,
        *,
        threshold: float,
    ) -> Tensor:
        """Return Base occupancy union CURE-Lite completion."""

        return (
            occupancy
            | self.predict_completion(
                feature,
                occupancy,
                threshold=threshold,
            )
        ).contiguous()


__all__ = [
    "CURELitePhaseResolvedRelationDecoder",
    "PFCR_EVIDENCE_CEILING",
    "PFCR_EVIDENCE_POLICY",
    "PFCR_INITIAL_BASELINE_PROBABILITY",
    "PFCR_OUTPUT_POLICY",
    "PFCR_RELEASE_POLICY",
    "PhaseResolvedRelationDecoderConfig",
    "PhaseResolvedRelationDecoderFields",
]
