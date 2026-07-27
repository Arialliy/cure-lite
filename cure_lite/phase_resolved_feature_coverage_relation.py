"""Phase-resolved feature--coverage relation for CURE-Lite.

This is the first model component of the independent CURE-Lite input
contract v2.  It replaces an undirected scalar occupancy count with one
shared relation state computed only from detached detector features ``F_b``
and binary Base occupancy ``O``.

For every output phase and every offset in a fixed 3 x 3 neighborhood, the
operator compares a feature-derived query at the candidate cell with a
feature-derived key at the occupied neighbor.  Occupancy gates the resulting
affinity.  The complete directional relation tensor is retained; the bounded
noisy-OR burden is only a deterministic summary for later state equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoder import project_occupancy_to_feature_grid


PFCR_RELATION_POLICY = (
    "phase_query_coverage_key_positive_cosine_v1"
)
PFCR_SMOOTH_RELATION_POLICY = (
    "phase_query_coverage_key_zero_preserving_softplus_cosine_v2"
)
PFCR_OCCUPANCY_POLICY = (
    "adaptive_max_then_directional_3x3_basis_v1"
)
PFCR_BURDEN_POLICY = "bounded_noisy_or_without_extra_parameters_v1"
PFCR_FEATURE_NORMALIZATION_POLICY = (
    "cellwise_zero_preserving_l2_before_shared_projection_v1"
)
PFCR_NEIGHBORHOOD_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class PhaseResolvedFeatureCoverageRelationConfig:
    """Frozen structural contract for the CURE-Lite relation state."""

    feature_channels: int
    feature_stride: int
    relation_dim: int = 8
    normalization_epsilon: float = 1.0e-6
    affinity_temperature: float = 4.0
    neighborhood_size: int = 3
    relation_policy: str = PFCR_RELATION_POLICY
    feature_normalization_policy: str = (
        PFCR_FEATURE_NORMALIZATION_POLICY
    )
    occupancy_policy: str = PFCR_OCCUPANCY_POLICY
    burden_policy: str = PFCR_BURDEN_POLICY

    def __post_init__(self) -> None:
        for name, value in (
            ("feature_channels", self.feature_channels),
            ("feature_stride", self.feature_stride),
            ("relation_dim", self.relation_dim),
            ("neighborhood_size", self.neighborhood_size),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.neighborhood_size != 3:
            raise ValueError("CURE-Lite relation v2 fixes a 3x3 basis")
        if (
            isinstance(self.normalization_epsilon, bool)
            or not isinstance(self.normalization_epsilon, float)
            or not 0.0 < self.normalization_epsilon < 1.0
        ):
            raise ValueError(
                "normalization_epsilon must be a float in (0,1)"
            )
        if (
            isinstance(self.affinity_temperature, bool)
            or not isinstance(self.affinity_temperature, float)
            or self.affinity_temperature != 4.0
        ):
            raise ValueError(
                "CURE-Lite relation v2 fixes affinity_temperature"
            )
        frozen = {
            "feature_normalization_policy": (
                PFCR_FEATURE_NORMALIZATION_POLICY
            ),
            "occupancy_policy": PFCR_OCCUPANCY_POLICY,
            "burden_policy": PFCR_BURDEN_POLICY,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"CURE-Lite relation v2 fixes {name}"
                )
        if self.relation_policy not in {
            PFCR_RELATION_POLICY,
            PFCR_SMOOTH_RELATION_POLICY,
        }:
            raise ValueError("unknown CURE-Lite relation policy")

    @property
    def phase_channels(self) -> int:
        return self.feature_stride**2

    @property
    def neighborhood_count(self) -> int:
        return self.neighborhood_size**2

    @property
    def projection_channels(self) -> int:
        return (self.phase_channels + 1) * self.relation_dim

    @property
    def expected_parameter_count(self) -> int:
        return self.feature_channels * self.projection_channels


@dataclass(frozen=True)
class PhaseResolvedFeatureCoverageRelationFields:
    """Every auditable state produced by the relation operator."""

    normalized_feature: Tensor
    phase_query: Tensor
    coverage_key: Tensor
    phase_evidence_strength: Tensor
    normalized_phase_query: Tensor
    normalized_coverage_key: Tensor
    projected_occupancy: Tensor
    occupancy_basis: Tensor
    affinity: Tensor
    relevant_coverage: Tensor
    coverage_burden: Tensor
    neighborhood_offsets: tuple[tuple[int, int], ...]


def zero_preserving_l2_normalize(
    value: Tensor,
    *,
    dim: int,
    epsilon: float,
) -> Tensor:
    """Normalize one tensor axis while mapping exact zero to exact zero."""

    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError("value must be a floating tensor")
    if (
        isinstance(dim, bool)
        or not isinstance(dim, int)
        or not -value.ndim <= dim < value.ndim
    ):
        raise ValueError("dim is outside the tensor rank")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, float)
        or not 0.0 < epsilon < 1.0
    ):
        raise ValueError("epsilon must be a float in (0,1)")
    norm = torch.linalg.vector_norm(
        value,
        ord=2,
        dim=dim,
        keepdim=True,
    )
    return value / norm.clamp_min(epsilon)


def directional_occupancy_basis(
    projected_occupancy: Tensor,
    *,
    neighborhood_size: int = 3,
) -> Tensor:
    """Return a row-major local occupancy basis ``[B,K,h,w]``."""

    if not isinstance(projected_occupancy, Tensor):
        raise TypeError("projected_occupancy must be a tensor")
    if (
        projected_occupancy.dtype != torch.bool
        or projected_occupancy.ndim != 4
        or projected_occupancy.shape[0] < 1
        or projected_occupancy.shape[1] != 1
        or min(projected_occupancy.shape[-2:]) < 1
    ):
        raise ValueError(
            "projected_occupancy must be bool [B,1,h,w]"
        )
    if neighborhood_size != 3:
        raise ValueError("the relation v2 basis fixes neighborhood_size=3")
    batch, _, height, width = projected_occupancy.shape
    unfolded = F.unfold(
        projected_occupancy.to(dtype=torch.float32),
        kernel_size=neighborhood_size,
        padding=neighborhood_size // 2,
    )
    basis = unfolded.reshape(
        batch,
        neighborhood_size**2,
        height,
        width,
    )
    return basis.to(dtype=torch.bool).contiguous()


class PhaseResolvedFeatureCoverageRelation(nn.Module):
    """One shared relation operator with no metadata or coordinate inputs."""

    config: PhaseResolvedFeatureCoverageRelationConfig

    def __init__(
        self,
        config: PhaseResolvedFeatureCoverageRelationConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(
            config,
            PhaseResolvedFeatureCoverageRelationConfig,
        ):
            if feature_channels is not None or feature_stride is not None:
                raise ValueError(
                    "do not override an explicit relation config"
                )
            resolved = config
        elif config is None:
            if feature_channels is None or feature_stride is None:
                raise TypeError(
                    "relation config or feature_channels/feature_stride "
                    "is required"
                )
            resolved = PhaseResolvedFeatureCoverageRelationConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
            )
        else:
            raise TypeError(
                "config must be "
                "PhaseResolvedFeatureCoverageRelationConfig or None"
            )
        self.config = resolved
        self.projection = nn.Conv2d(
            resolved.feature_channels,
            resolved.projection_channels,
            kernel_size=1,
            bias=False,
        )
        self._reset_parameters()
        actual = sum(
            parameter.numel() for parameter in self.parameters()
        )
        if actual != resolved.expected_parameter_count:
            raise AssertionError(
                "relation parameter count differs from its contract"
            )

    def _reset_parameters(self) -> None:
        if self.config.relation_policy == PFCR_RELATION_POLICY:
            nn.init.xavier_normal_(self.projection.weight, gain=1.0)
            return
        if (
            self.config.relation_policy
            != PFCR_SMOOTH_RELATION_POLICY
        ):
            raise AssertionError("unreachable relation policy")
        with torch.no_grad():
            self.projection.weight.zero_()
            key_offset = (
                self.config.phase_channels * self.config.relation_dim
            )
            key_weight = self.projection.weight[
                key_offset : key_offset + self.config.relation_dim
            ]
            nn.init.xavier_normal_(key_weight, gain=1.0)
            for phase_index in range(self.config.phase_channels):
                start = phase_index * self.config.relation_dim
                self.projection.weight[
                    start : start + self.config.relation_dim
                ].copy_(key_weight)

    def _validate_inputs(
        self,
        feature: Tensor,
        occupancy: Tensor,
        *,
        check_finite: bool = True,
    ) -> None:
        if not isinstance(feature, Tensor) or not isinstance(
            occupancy,
            Tensor,
        ):
            raise TypeError("feature and occupancy must be tensors")
        if (
            feature.ndim != 4
            or feature.shape[0] < 1
            or feature.shape[1] != self.config.feature_channels
            or min(feature.shape[-2:]) < 1
        ):
            raise ValueError(
                "feature must be nonempty [B,C,h,w] with configured C"
            )
        if (
            occupancy.ndim != 4
            or occupancy.shape[0] != feature.shape[0]
            or occupancy.shape[1] != 1
            or min(occupancy.shape[-2:]) < 1
        ):
            raise ValueError(
                "occupancy must be batch-aligned [B,1,H,W]"
            )
        if not feature.is_floating_point():
            raise TypeError("feature must be floating point")
        if check_finite and not bool(torch.isfinite(feature).all()):
            raise ValueError("feature must be finite")
        if feature.dtype != self.projection.weight.dtype:
            raise TypeError(
                "feature dtype must match relation parameters"
            )
        if occupancy.dtype != torch.bool:
            raise TypeError("occupancy must be bool")
        if feature.device != occupancy.device:
            raise ValueError("feature and occupancy must share a device")
        if feature.device != self.projection.weight.device:
            raise ValueError(
                "inputs and relation parameters must share a device"
            )
        if any(
            feature_size > output_size
            for feature_size, output_size in zip(
                feature.shape[-2:],
                occupancy.shape[-2:],
                strict=True,
            )
        ):
            raise ValueError("occupancy projection may not upsample")

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> PhaseResolvedFeatureCoverageRelationFields:
        """Build the complete directional relation state from ``(F_b,O)``."""

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
    ) -> PhaseResolvedFeatureCoverageRelationFields:
        """Execute one equation, optionally omitting repeated tensor scans."""

        if not isinstance(audit, bool):
            raise TypeError("audit must be bool")
        self._validate_inputs(
            feature,
            occupancy,
            check_finite=audit,
        )
        batch, _, height, width = feature.shape
        normalized_feature = zero_preserving_l2_normalize(
            feature.detach(),
            dim=1,
            epsilon=self.config.normalization_epsilon,
        )
        projected = self.projection(normalized_feature)
        projected = projected.reshape(
            batch,
            self.config.phase_channels + 1,
            self.config.relation_dim,
            height,
            width,
        )
        phase_query = projected[:, : self.config.phase_channels]
        coverage_key = projected[:, self.config.phase_channels]
        normalized_query = zero_preserving_l2_normalize(
            phase_query,
            dim=2,
            epsilon=self.config.normalization_epsilon,
        )
        normalized_key = zero_preserving_l2_normalize(
            coverage_key,
            dim=1,
            epsilon=self.config.normalization_epsilon,
        )
        phase_evidence_strength = torch.linalg.vector_norm(
            phase_query,
            ord=2,
            dim=2,
        )

        projected_occupancy = project_occupancy_to_feature_grid(
            occupancy,
            (int(height), int(width)),
        )
        occupancy_basis = directional_occupancy_basis(
            projected_occupancy,
            neighborhood_size=self.config.neighborhood_size,
        )
        unfolded_key = F.unfold(
            normalized_key,
            kernel_size=self.config.neighborhood_size,
            padding=self.config.neighborhood_size // 2,
        ).reshape(
            batch,
            self.config.relation_dim,
            self.config.neighborhood_count,
            height,
            width,
        )
        cosine = torch.einsum(
            "bpdhw,bdkhw->bpkhw",
            normalized_query,
            unfolded_key,
        ).clamp(min=-1.0, max=1.0)
        if self.config.relation_policy == PFCR_RELATION_POLICY:
            affinity = cosine.clamp(min=0.0, max=1.0)
        elif (
            self.config.relation_policy
            == PFCR_SMOOTH_RELATION_POLICY
        ):
            temperature = self.config.affinity_temperature
            denominator = F.softplus(
                torch.tensor(
                    temperature,
                    dtype=cosine.dtype,
                    device=cosine.device,
                )
            )
            smooth = F.softplus(temperature * cosine) / denominator
            query_active = (
                torch.linalg.vector_norm(
                    normalized_query,
                    ord=2,
                    dim=2,
                )
                > 0.0
            )
            key_active = (
                torch.linalg.vector_norm(
                    unfolded_key,
                    ord=2,
                    dim=1,
                )
                > 0.0
            )
            affinity = (
                smooth
                * query_active[:, :, None].to(dtype=smooth.dtype)
                * key_active[:, None].to(dtype=smooth.dtype)
            ).clamp(min=0.0, max=1.0)
        else:
            raise AssertionError("unreachable relation policy")
        relevant_coverage = affinity * occupancy_basis[:, None].to(
            dtype=affinity.dtype
        )
        coverage_burden = (
            1.0 - torch.prod(1.0 - relevant_coverage, dim=2)
        ).clamp(min=0.0, max=1.0)

        fields = PhaseResolvedFeatureCoverageRelationFields(
            normalized_feature=normalized_feature.contiguous(),
            phase_query=phase_query.contiguous(),
            coverage_key=coverage_key.contiguous(),
            phase_evidence_strength=(
                phase_evidence_strength.contiguous()
            ),
            normalized_phase_query=normalized_query.contiguous(),
            normalized_coverage_key=normalized_key.contiguous(),
            projected_occupancy=projected_occupancy.contiguous(),
            occupancy_basis=occupancy_basis,
            affinity=affinity.contiguous(),
            relevant_coverage=relevant_coverage.contiguous(),
            coverage_burden=coverage_burden.contiguous(),
            neighborhood_offsets=PFCR_NEIGHBORHOOD_OFFSETS,
        )
        if audit:
            self._validate_fields(fields, feature=feature)
        return fields

    def _validate_fields(
        self,
        fields: PhaseResolvedFeatureCoverageRelationFields,
        *,
        feature: Tensor,
    ) -> None:
        batch, _, height, width = feature.shape
        phase_shape = (
            batch,
            self.config.phase_channels,
            self.config.relation_dim,
            height,
            width,
        )
        key_shape = (
            batch,
            self.config.relation_dim,
            height,
            width,
        )
        relation_shape = (
            batch,
            self.config.phase_channels,
            self.config.neighborhood_count,
            height,
            width,
        )
        burden_shape = (
            batch,
            self.config.phase_channels,
            height,
            width,
        )
        expected = (
            (
                "normalized_feature",
                fields.normalized_feature,
                tuple(feature.shape),
            ),
            ("phase_query", fields.phase_query, phase_shape),
            (
                "normalized_phase_query",
                fields.normalized_phase_query,
                phase_shape,
            ),
            ("coverage_key", fields.coverage_key, key_shape),
            (
                "phase_evidence_strength",
                fields.phase_evidence_strength,
                burden_shape,
            ),
            (
                "normalized_coverage_key",
                fields.normalized_coverage_key,
                key_shape,
            ),
            ("affinity", fields.affinity, relation_shape),
            (
                "relevant_coverage",
                fields.relevant_coverage,
                relation_shape,
            ),
            (
                "coverage_burden",
                fields.coverage_burden,
                burden_shape,
            ),
        )
        for name, value, shape in expected:
            if tuple(value.shape) != shape:
                raise AssertionError(f"{name} has an invalid shape")
            if value.dtype != feature.dtype or value.device != feature.device:
                raise ValueError(
                    f"{name} must match feature dtype/device"
                )
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"{name} must be finite")
        occupancy_shape = (batch, 1, height, width)
        basis_shape = (
            batch,
            self.config.neighborhood_count,
            height,
            width,
        )
        if (
            tuple(fields.projected_occupancy.shape) != occupancy_shape
            or fields.projected_occupancy.dtype != torch.bool
        ):
            raise AssertionError(
                "projected_occupancy has an invalid contract"
            )
        if (
            tuple(fields.occupancy_basis.shape) != basis_shape
            or fields.occupancy_basis.dtype != torch.bool
        ):
            raise AssertionError(
                "occupancy_basis has an invalid contract"
            )
        for name, value in (
            ("affinity", fields.affinity),
            ("relevant_coverage", fields.relevant_coverage),
            ("coverage_burden", fields.coverage_burden),
        ):
            if not bool(((value >= 0.0) & (value <= 1.0)).all()):
                raise AssertionError(f"{name} must be bounded in [0,1]")
        feature_norm = torch.linalg.vector_norm(
            fields.normalized_feature,
            ord=2,
            dim=1,
        )
        if not bool((feature_norm <= 1.0 + 1.0e-6).all()):
            raise AssertionError(
                "normalized feature cell norms must be bounded by one"
            )

    def forward(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        """Return the bounded phase-specific coverage burden."""

        return self.forward_fields(feature, occupancy).coverage_burden


__all__ = [
    "PFCR_BURDEN_POLICY",
    "PFCR_FEATURE_NORMALIZATION_POLICY",
    "PFCR_NEIGHBORHOOD_OFFSETS",
    "PFCR_OCCUPANCY_POLICY",
    "PFCR_RELATION_POLICY",
    "PFCR_SMOOTH_RELATION_POLICY",
    "PhaseResolvedFeatureCoverageRelation",
    "PhaseResolvedFeatureCoverageRelationConfig",
    "PhaseResolvedFeatureCoverageRelationFields",
    "directional_occupancy_basis",
    "zero_preserving_l2_normalize",
]
