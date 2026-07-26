"""Null-Anchored Local Count Crossing for CURE-Lite v12.

NLCC keeps the active-v4 module topology unchanged.  The shared trunk and
baseline are feature-only.  Occupancy enters once, after every convolution
and spatial normalization, as a unit local-count boundary in one
null-anchored phase crossing equation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .decoder import project_occupancy_to_feature_grid
from .factorized_decoder import CURELiteFactorizedDecoder
from .null_anchored_local_count_crossing_config import (
    NullAnchoredLocalCountCrossingDecoderConfig,
)


@dataclass(frozen=True)
class NullAnchoredLocalCountCrossingDecoderFields:
    """Auditable fields produced by one NLCC forward."""

    stem_feature: Tensor
    trunk_feature: Tensor
    baseline_logits: Tensor
    raw_phase_evidence: Tensor
    null_anchored_reference: Tensor
    phase_relative_evidence: Tensor
    count_boundary: Tensor
    crossing_margin: Tensor
    active_phase_mask: Tensor
    recovery_factor: Tensor
    native_phase_evidence: Tensor
    evidence: Tensor
    logits: Tensor
    projected_occupancy: Tensor
    local_occupancy_count: Tensor
    native_subpixel_size: tuple[int, int]
    output_size: tuple[int, int]
    field_resize_applied: bool


def null_anchored_local_count_crossing(
    raw_phase_evidence: Tensor,
    local_occupancy_count: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Apply the frozen NLCC joint state equation.

    For ``P`` raw phase responses and one fixed zero null coordinate,

    ``reference = sum(raw) / (P + 1)``
    ``margin = raw - reference - count``.

    The observable forward evidence is zero for a nonpositive margin and
    ``expm1(margin)`` otherwise.  The recovery carrier is ``exp(margin)``
    on the complete supported numeric axis.  No parameter, clamp,
    temperature, or trainable count scale is introduced.
    """

    if not isinstance(raw_phase_evidence, Tensor):
        raise TypeError("raw_phase_evidence must be a tensor")
    if not isinstance(local_occupancy_count, Tensor):
        raise TypeError("local_occupancy_count must be a tensor")
    if (
        raw_phase_evidence.ndim != 4
        or raw_phase_evidence.shape[0] < 1
        or raw_phase_evidence.shape[1] < 1
        or min(raw_phase_evidence.shape[-2:]) < 1
    ):
        raise ValueError(
            "raw_phase_evidence must have shape [B,P,h,w]"
        )
    expected_count_shape = (
        int(raw_phase_evidence.shape[0]),
        1,
        int(raw_phase_evidence.shape[2]),
        int(raw_phase_evidence.shape[3]),
    )
    if tuple(local_occupancy_count.shape) != expected_count_shape:
        raise ValueError(
            "local_occupancy_count must have shape [B,1,h,w]"
        )
    if (
        not raw_phase_evidence.is_floating_point()
        or not local_occupancy_count.is_floating_point()
    ):
        raise TypeError("NLCC fields must be floating point")
    if raw_phase_evidence.dtype != local_occupancy_count.dtype:
        raise TypeError("NLCC fields must share a dtype")
    if raw_phase_evidence.device != local_occupancy_count.device:
        raise ValueError("NLCC fields must share a device")
    input_contract = torch.stack(
        (
            torch.isfinite(raw_phase_evidence).all(),
            torch.isfinite(local_occupancy_count).all(),
            (local_occupancy_count >= 0.0).all(),
            (local_occupancy_count <= 9.0).all(),
            (
                local_occupancy_count
                == torch.round(local_occupancy_count)
            ).all(),
        )
    )
    if not bool(input_contract.all()):
        if not bool(input_contract[0]):
            raise FloatingPointError(
                "raw_phase_evidence must be finite"
            )
        if not bool(input_contract[1]):
            raise FloatingPointError(
                "local_occupancy_count must be finite"
            )
        raise ValueError(
            "local_occupancy_count must contain integers in [0,9]"
        )

    phase_count = int(raw_phase_evidence.shape[1])
    null_reference = (
        raw_phase_evidence.sum(dim=1, keepdim=True)
        / float(phase_count + 1)
    )
    phase_relative = raw_phase_evidence - null_reference
    count_boundary = local_occupancy_count
    crossing_margin = phase_relative - count_boundary
    continuation = torch.expm1(crossing_margin)
    recovery_factor = torch.exp(crossing_margin)
    operator_finite = torch.stack(
        (
            torch.isfinite(crossing_margin).all(),
            torch.isfinite(continuation).all(),
            torch.isfinite(recovery_factor).all(),
        )
    )
    if not bool(operator_finite.all()):
        if not bool(operator_finite[0]):
            raise FloatingPointError(
                "NLCC crossing margin must be finite"
            )
        raise FloatingPointError(
            "NLCC evidence and recovery factor must be finite"
        )
    active_phase_mask = crossing_margin > 0.0
    forward_evidence = torch.where(
        active_phase_mask,
        continuation,
        torch.zeros_like(continuation),
    )
    native_phase_evidence = forward_evidence.detach() + (
        recovery_factor - recovery_factor.detach()
    )
    return (
        null_reference,
        phase_relative,
        count_boundary,
        crossing_margin,
        active_phase_mask,
        recovery_factor,
        native_phase_evidence,
    )


class CURELiteNullAnchoredLocalCountCrossingDecoder(
    CURELiteFactorizedDecoder
):
    """The active-v4 topology with the single NLCC-v12 equation."""

    config: NullAnchoredLocalCountCrossingDecoderConfig

    def __init__(
        self,
        config: NullAnchoredLocalCountCrossingDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        if isinstance(
            config,
            NullAnchoredLocalCountCrossingDecoderConfig,
        ):
            if feature_channels is not None or feature_stride is not None:
                raise ValueError(
                    "do not override an explicit "
                    "NullAnchoredLocalCountCrossingDecoderConfig"
                )
            resolved = config
        elif config is None:
            if feature_channels is None or feature_stride is None:
                raise TypeError(
                    "NullAnchoredLocalCountCrossingDecoderConfig or "
                    "feature_channels/feature_stride is required"
                )
            resolved = NullAnchoredLocalCountCrossingDecoderConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
            )
        else:
            raise TypeError(
                "config must be "
                "NullAnchoredLocalCountCrossingDecoderConfig or None"
            )

        super().__init__(resolved.to_v4_topology_config())
        self.config = resolved
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != resolved.expected_parameter_count:
            raise AssertionError(
                "NLCC parameter count differs from the frozen topology"
            )

    def native_count_field(
        self,
        occupancy: Tensor,
        *,
        feature_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        """Return max-projected occupancy and its native 3x3 count."""

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
        return projected.contiguous(), count.contiguous()

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> NullAnchoredLocalCountCrossingDecoderFields:
        """Return every NLCC field without changing public inputs."""

        output_size = self._validate_inputs(feature, occupancy)
        detached = feature.detach()
        stem_feature = F.silu(
            self.stem_norm(self.stem(detached))
        )
        residual = self.pointwise(
            F.silu(
                self.depthwise_norm(
                    self.depthwise(stem_feature)
                )
            )
        )
        trunk_feature = (
            stem_feature
            + self.config.trunk_residual_scale * residual
        )

        baseline_phase = self.baseline_head(trunk_feature)
        raw_phase_evidence = self.evidence_head(trunk_feature)
        projected, count = self.native_count_field(
            occupancy,
            feature_size=tuple(
                int(value) for value in feature.shape[-2:]
            ),
        )
        (
            null_reference,
            phase_relative,
            count_boundary,
            crossing_margin,
            active_phase_mask,
            recovery_factor,
            native_phase_evidence,
        ) = null_anchored_local_count_crossing(
            raw_phase_evidence,
            count,
        )

        baseline_native = self.pixel_shuffle(baseline_phase)
        evidence_native = self.pixel_shuffle(native_phase_evidence)
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
        if not bool(
            torch.stack(
                (
                    torch.isfinite(stem_feature).all(),
                    torch.isfinite(trunk_feature).all(),
                    torch.isfinite(baseline_logits).all(),
                    torch.isfinite(evidence).all(),
                    torch.isfinite(logits).all(),
                )
            ).all()
        ):
            raise FloatingPointError(
                "NLCC forward fields must all be finite"
            )
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
        native_phase_shape = (
            int(feature.shape[0]),
            self.config.phase_channels,
            int(feature.shape[-2]),
            int(feature.shape[-1]),
        )
        native_count_shape = (
            int(feature.shape[0]),
            1,
            int(feature.shape[-2]),
            int(feature.shape[-1]),
        )
        for name, value, expected_shape in (
            ("stem_feature", stem_feature, native_feature_shape),
            ("trunk_feature", trunk_feature, native_feature_shape),
            (
                "raw_phase_evidence",
                raw_phase_evidence,
                native_phase_shape,
            ),
            (
                "null_anchored_reference",
                null_reference,
                native_count_shape,
            ),
            (
                "phase_relative_evidence",
                phase_relative,
                native_phase_shape,
            ),
            ("count_boundary", count_boundary, native_count_shape),
            ("crossing_margin", crossing_margin, native_phase_shape),
            ("recovery_factor", recovery_factor, native_phase_shape),
            (
                "native_phase_evidence",
                native_phase_evidence,
                native_phase_shape,
            ),
        ):
            if tuple(value.shape) != expected_shape:
                raise AssertionError(f"{name} has an invalid shape")
            if value.dtype != feature.dtype or value.device != feature.device:
                raise ValueError(
                    f"{name} must match the feature dtype/device"
                )
        if tuple(active_phase_mask.shape) != native_phase_shape:
            raise AssertionError(
                "active_phase_mask has an invalid shape"
            )
        if (
            active_phase_mask.dtype != torch.bool
            or active_phase_mask.device != feature.device
        ):
            raise TypeError(
                "active_phase_mask must be a device-aligned bool tensor"
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

        return NullAnchoredLocalCountCrossingDecoderFields(
            stem_feature=stem_feature,
            trunk_feature=trunk_feature,
            baseline_logits=baseline_logits,
            raw_phase_evidence=raw_phase_evidence,
            null_anchored_reference=null_reference,
            phase_relative_evidence=phase_relative,
            count_boundary=count_boundary,
            crossing_margin=crossing_margin,
            active_phase_mask=active_phase_mask,
            recovery_factor=recovery_factor,
            native_phase_evidence=native_phase_evidence,
            evidence=evidence,
            logits=logits,
            projected_occupancy=projected,
            local_occupancy_count=count,
            native_subpixel_size=native_size,
            output_size=output_size,
            field_resize_applied=resize,
        )


__all__ = [
    "CURELiteNullAnchoredLocalCountCrossingDecoder",
    "NullAnchoredLocalCountCrossingDecoderFields",
    "null_anchored_local_count_crossing",
]
