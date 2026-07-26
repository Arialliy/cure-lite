"""Coverage-Conditioned Feature Release for CURE-Lite v11.

CCFR keeps the frozen v4 module topology and moves the fixed inverse-count
coverage field from a post-head evidence multiplier into the shared trunk.
Coverage therefore changes the representation consumed by the existing
depthwise spatial transform and heads instead of only scaling an already
formed evidence field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .coverage_feature_release_config import (
    CoverageFeatureReleaseDecoderConfig,
)
from .decoder import project_occupancy_to_feature_grid
from .factorized_decoder import CURELiteFactorizedDecoder


@dataclass(frozen=True)
class CoverageFeatureReleaseDecoderFields:
    """Auditable fields produced by one CCFR forward."""

    stem_feature: Tensor
    feature_release: Tensor
    released_stem_feature: Tensor
    trunk_feature: Tensor
    baseline_logits: Tensor
    raw_phase_evidence: Tensor
    common_mode_phase_evidence: Tensor
    budget_margin: Tensor
    evidence_budget: Tensor
    phase_allocation: Tensor
    allocated_phase_evidence: Tensor
    evidence: Tensor
    logits: Tensor
    projected_occupancy: Tensor
    local_occupancy_count: Tensor
    native_subpixel_size: tuple[int, int]
    output_size: tuple[int, int]
    field_resize_applied: bool


def occupancy_free_conserving_phase_evidence(
    raw_phase_evidence: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return the v8 phase operator with zero output-side burden.

    This is algebraically identical to
    ``coverage_conserving_phase_evidence(raw, zeros)``.  It deliberately
    performs only metadata validation in the decoder hot path; the paired
    training step and protocol audits already reject nonfinite losses and
    gradients.  Avoiding repeated tensor-wide Python reductions prevents a
    device synchronization on every decoder forward.
    """

    if not isinstance(raw_phase_evidence, Tensor):
        raise TypeError("raw_phase_evidence must be a tensor")
    if (
        raw_phase_evidence.ndim != 4
        or raw_phase_evidence.shape[0] < 1
        or raw_phase_evidence.shape[1] < 1
        or min(raw_phase_evidence.shape[-2:]) < 1
    ):
        raise ValueError(
            "raw_phase_evidence must have shape [B,P,h,w]"
        )
    if not raw_phase_evidence.is_floating_point():
        raise TypeError("raw_phase_evidence must be floating point")

    common_mode = raw_phase_evidence.mean(dim=1, keepdim=True)
    phase_contrast = raw_phase_evidence - common_mode
    continuation = torch.expm1(common_mode)
    recovery = torch.exp(common_mode)
    forward_budget = torch.where(
        common_mode <= 0.0,
        torch.zeros_like(continuation),
        continuation,
    )
    evidence_budget = forward_budget.detach() + (
        recovery - recovery.detach()
    )
    phase_allocation = torch.softmax(phase_contrast, dim=1)
    allocated = phase_allocation * evidence_budget
    return (
        common_mode,
        common_mode,
        evidence_budget,
        phase_allocation,
        allocated,
    )


class CURELiteCoverageFeatureReleaseDecoder(
    CURELiteFactorizedDecoder
):
    """The v4 topology with one joint feature-coverage state."""

    config: CoverageFeatureReleaseDecoderConfig

    def __init__(
        self,
        config: CoverageFeatureReleaseDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        if isinstance(config, CoverageFeatureReleaseDecoderConfig):
            if feature_channels is not None or feature_stride is not None:
                raise ValueError(
                    "do not override an explicit "
                    "CoverageFeatureReleaseDecoderConfig"
                )
            resolved = config
        elif config is None:
            if feature_channels is None or feature_stride is None:
                raise TypeError(
                    "CoverageFeatureReleaseDecoderConfig or "
                    "feature_channels/feature_stride is required"
                )
            resolved = CoverageFeatureReleaseDecoderConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
            )
        else:
            raise TypeError(
                "config must be "
                "CoverageFeatureReleaseDecoderConfig or None"
            )

        super().__init__(resolved.to_v4_topology_config())
        self.config = resolved
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != resolved.expected_parameter_count:
            raise AssertionError(
                "CCFR parameter count differs from the frozen topology"
            )

    def native_release_field(
        self,
        occupancy: Tensor,
        *,
        feature_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return projected occupancy, local count, and native release."""

        if not isinstance(occupancy, Tensor):
            raise TypeError("occupancy must be a tensor")
        if (
            occupancy.ndim != 4
            or occupancy.shape[0] < 1
            or occupancy.shape[1] != 1
            or min(occupancy.shape[-2:]) < 1
        ):
            raise ValueError("occupancy must have shape [B,1,H,W]")
        if occupancy.dtype != torch.bool:
            raise TypeError("occupancy must be bool")
        if (
            not isinstance(feature_size, tuple)
            or len(feature_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in feature_size
            )
        ):
            raise ValueError(
                "feature_size must contain two positive integers"
            )
        if occupancy.device != self._vacancy_kernel.device:
            raise ValueError("occupancy and decoder must share a device")

        projected = project_occupancy_to_feature_grid(
            occupancy,
            feature_size,
        )
        count = F.conv2d(
            projected.to(dtype=self._vacancy_kernel.dtype),
            self._vacancy_kernel,
            padding=self.config.vacancy_kernel_size // 2,
        )
        release = count.add(1.0).reciprocal()
        return (
            projected.contiguous(),
            count.contiguous(),
            release.contiguous(),
        )

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CoverageFeatureReleaseDecoderFields:
        """Return all CCFR fields without changing the public inputs."""

        output_size = self._validate_inputs(feature, occupancy)
        detached = feature.detach()
        stem_feature = F.silu(
            self.stem_norm(self.stem(detached))
        )
        projected, count, release = self.native_release_field(
            occupancy,
            feature_size=tuple(
                int(value) for value in feature.shape[-2:]
            ),
        )
        released_stem = stem_feature * release
        residual = self.pointwise(
            F.silu(
                self.depthwise_norm(
                    self.depthwise(released_stem)
                )
            )
        )
        trunk = (
            released_stem
            + self.config.trunk_residual_scale * residual
        )

        baseline_phase = self.baseline_head(trunk)
        raw_phase_evidence = self.evidence_head(trunk)
        (
            common_mode,
            budget_margin,
            evidence_budget,
            phase_allocation,
            allocated_phase_evidence,
        ) = occupancy_free_conserving_phase_evidence(
            raw_phase_evidence,
        )
        baseline_native = self.pixel_shuffle(baseline_phase)
        evidence_native = self.pixel_shuffle(
            allocated_phase_evidence
        )
        native_size = tuple(
            int(value) for value in baseline_native.shape[-2:]
        )
        resize = native_size != output_size
        if resize:
            baseline_raw = F.interpolate(
                baseline_native,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            evidence = F.interpolate(
                evidence_native,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        else:
            baseline_raw = baseline_native
            evidence = evidence_native

        baseline_logits = -F.softplus(
            self.baseline_raw.reshape(1, 1, 1, 1)
            + baseline_raw
        )
        logits = baseline_logits + evidence

        output_shape = (
            int(feature.shape[0]),
            1,
            int(output_size[0]),
            int(output_size[1]),
        )
        native_feature_shape = (
            int(feature.shape[0]),
            self.config.width,
            int(feature.shape[-2]),
            int(feature.shape[-1]),
        )
        for name, value in (
            ("stem_feature", stem_feature),
            ("released_stem_feature", released_stem),
            ("trunk_feature", trunk),
        ):
            if tuple(value.shape) != native_feature_shape:
                raise AssertionError(f"{name} has an invalid shape")
            if value.dtype != feature.dtype or value.device != feature.device:
                raise ValueError(
                    f"{name} must match the feature dtype/device"
                )
        for name, value in (
            ("baseline_logits", baseline_logits),
            ("evidence", evidence),
            ("logits", logits),
        ):
            if tuple(value.shape) != output_shape:
                raise AssertionError(f"{name} has an invalid shape")
            if value.dtype != feature.dtype or value.device != feature.device:
                raise ValueError(
                    f"{name} must match the feature dtype/device"
                )

        return CoverageFeatureReleaseDecoderFields(
            stem_feature=stem_feature,
            feature_release=release,
            released_stem_feature=released_stem,
            trunk_feature=trunk,
            baseline_logits=baseline_logits,
            raw_phase_evidence=raw_phase_evidence,
            common_mode_phase_evidence=common_mode,
            budget_margin=budget_margin,
            evidence_budget=evidence_budget,
            phase_allocation=phase_allocation,
            allocated_phase_evidence=allocated_phase_evidence,
            evidence=evidence,
            logits=logits,
            projected_occupancy=projected,
            local_occupancy_count=count,
            native_subpixel_size=native_size,
            output_size=output_size,
            field_resize_applied=resize,
        )


__all__ = [
    "CURELiteCoverageFeatureReleaseDecoder",
    "CoverageFeatureReleaseDecoderFields",
    "occupancy_free_conserving_phase_evidence",
]
