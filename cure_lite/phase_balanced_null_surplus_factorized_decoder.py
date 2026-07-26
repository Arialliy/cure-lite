"""Phase-Balanced Null-Anchored Evidence Surplus for CURE-Lite v9.

PB-NAES changes only the evidence equation of the frozen v4 decoder.  The
shared trunk, two existing heads, parameter count, PixelShuffle path,
baseline field, and final single-pass composition remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .decoder import project_occupancy_to_feature_grid
from .factorized_decoder import CURELiteFactorizedDecoder
from .phase_balanced_null_surplus_factorized_config import (
    PhaseBalancedNullSurplusFactorizedDecoderConfig,
)


@dataclass(frozen=True)
class PhaseBalancedNullSurplusFactorizedDecoderFields:
    """Auditable fields produced by one PB-NAES forward."""

    baseline_logits: Tensor
    raw_phase_evidence: Tensor
    phase_intensity: Tensor
    implicit_null_threshold: Tensor
    signed_phase_surplus: Tensor
    active_phase_mask: Tensor
    native_phase_evidence: Tensor
    evidence: Tensor
    logits: Tensor
    projected_occupancy: Tensor
    local_occupancy_count: Tensor
    native_subpixel_size: tuple[int, int]
    output_size: tuple[int, int]
    field_resize_applied: bool


def phase_balanced_null_surplus_evidence(
    raw_phase_evidence: Tensor,
    local_occupancy_count: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return the parameter-free PB-NAES phase evidence.

    For ``P`` phase responses ``r_j``, phase intensities ``x_j=exp(r_j)``,
    and the native 3x3 projected-occupancy count ``C``, PB-NAES defines

    ``m = (P + sum_j(x_j)) / (2P)``,
    ``s_j = (x_j - m) / (1 + C)``,
    ``e_j = max(s_j, 0)`` in the observable forward pass.

    The fixed term ``P`` is one unit-intensity implicit null for every phase:
    total phase and total null prior mass are therefore balanced 1:1.  The
    backward path is the complete signed surplus ``s_j`` rather than the
    derivative of the hard positive selection:

    ``e = stopgrad(max(s,0)) + s - stopgrad(s)``.

    Thus inactive phases retain a recovery direction, while the observable
    evidence is exactly nonnegative and zero at the uniform ``r=0`` anchor.
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
        raise TypeError("PB-NAES fields must be floating point")
    if raw_phase_evidence.dtype != local_occupancy_count.dtype:
        raise TypeError("PB-NAES fields must share a dtype")
    if raw_phase_evidence.device != local_occupancy_count.device:
        raise ValueError("PB-NAES fields must share a device")
    if not torch.isfinite(raw_phase_evidence).all():
        raise ValueError("raw_phase_evidence must be finite")
    if not torch.isfinite(local_occupancy_count).all():
        raise ValueError("local_occupancy_count must be finite")
    if torch.any(local_occupancy_count < 0.0):
        raise ValueError(
            "local_occupancy_count must be nonnegative"
        )

    phase_count = int(raw_phase_evidence.shape[1])
    phase_intensity = torch.exp(raw_phase_evidence)
    if not torch.isfinite(phase_intensity).all():
        raise ValueError("phase intensity must remain finite")
    implicit_null_threshold = (
        phase_intensity.sum(dim=1, keepdim=True) + phase_count
    ) / (2 * phase_count)
    signed_phase_surplus = (
        phase_intensity - implicit_null_threshold
    ) / (local_occupancy_count + 1.0)
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            implicit_null_threshold,
            signed_phase_surplus,
        )
    ):
        raise ValueError("PB-NAES operator produced a nonfinite field")

    active_phase_mask = signed_phase_surplus > 0.0
    forward_evidence = torch.where(
        active_phase_mask,
        signed_phase_surplus,
        torch.zeros_like(signed_phase_surplus),
    )
    native_phase_evidence = forward_evidence.detach() + (
        signed_phase_surplus - signed_phase_surplus.detach()
    )
    return (
        phase_intensity,
        implicit_null_threshold,
        signed_phase_surplus,
        active_phase_mask,
        native_phase_evidence,
    )


class CURELitePhaseBalancedNullSurplusFactorizedDecoder(
    CURELiteFactorizedDecoder
):
    """The unchanged v4 topology with the single PB-NAES v9 equation."""

    config: PhaseBalancedNullSurplusFactorizedDecoderConfig

    def __init__(
        self,
        config: (
            PhaseBalancedNullSurplusFactorizedDecoderConfig | None
        ) = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        if isinstance(
            config,
            PhaseBalancedNullSurplusFactorizedDecoderConfig,
        ):
            if feature_channels is not None or feature_stride is not None:
                raise ValueError(
                    "do not override an explicit "
                    "PhaseBalancedNullSurplusFactorizedDecoderConfig"
                )
            resolved = config
        elif config is None:
            if feature_channels is None or feature_stride is None:
                raise TypeError(
                    "PhaseBalancedNullSurplusFactorizedDecoderConfig or "
                    "feature_channels/feature_stride is required"
                )
            resolved = PhaseBalancedNullSurplusFactorizedDecoderConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
            )
        else:
            raise TypeError(
                "config must be "
                "PhaseBalancedNullSurplusFactorizedDecoderConfig or None"
            )

        super().__init__(resolved.to_v4_topology_config())
        self.config = resolved
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != resolved.expected_parameter_count:
            raise AssertionError(
                "PB-NAES parameter count differs from the frozen topology"
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
    ) -> PhaseBalancedNullSurplusFactorizedDecoderFields:
        """Return every PB-NAES field without changing the v4 topology."""

        output_size = self._validate_inputs(feature, occupancy)
        detached = feature.detach()
        trunk0 = F.silu(self.stem_norm(self.stem(detached)))
        residual = self.pointwise(
            F.silu(self.depthwise_norm(self.depthwise(trunk0)))
        )
        trunk = trunk0 + self.config.trunk_residual_scale * residual

        baseline_phase = self.baseline_head(trunk)
        raw_phase_evidence = self.evidence_head(trunk)
        projected, count = self.native_count_field(
            occupancy,
            feature_size=tuple(
                int(value) for value in feature.shape[-2:]
            ),
        )
        (
            phase_intensity,
            implicit_null_threshold,
            signed_phase_surplus,
            active_phase_mask,
            native_phase_evidence,
        ) = phase_balanced_null_surplus_evidence(
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
        output_shape = (
            int(feature.shape[0]),
            1,
            int(output_size[0]),
            int(output_size[1]),
        )
        for name, value in (
            ("baseline_logits", baseline_logits),
            ("evidence", evidence),
            ("logits", logits),
        ):
            if tuple(value.shape) != output_shape:
                raise AssertionError(f"{name} has an invalid shape")
            if value.dtype != feature.dtype or value.device != feature.device:
                raise AssertionError(
                    f"{name} differs from the feature dtype/device"
                )

        return PhaseBalancedNullSurplusFactorizedDecoderFields(
            baseline_logits=baseline_logits,
            raw_phase_evidence=raw_phase_evidence,
            phase_intensity=phase_intensity,
            implicit_null_threshold=implicit_null_threshold,
            signed_phase_surplus=signed_phase_surplus,
            active_phase_mask=active_phase_mask,
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
    "CURELitePhaseBalancedNullSurplusFactorizedDecoder",
    "PhaseBalancedNullSurplusFactorizedDecoderFields",
    "phase_balanced_null_surplus_evidence",
]
