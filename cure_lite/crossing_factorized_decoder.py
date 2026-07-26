"""Crossing-recoverable local vacancy evidence for CURE-Lite v7.

CR-LVEC keeps the v4 decoder topology and replaces reciprocal-vacancy
modulation with one exponential-ratio crossing equation.  Occupancy therefore
changes logits only where its projected 3x3 count changes, without computing
the discarded reciprocal vacancy field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .crossing_factorized_config import (
    CrossingFactorizedDecoderConfig,
)
from .decoder import project_occupancy_to_feature_grid
from .factorized_decoder import CURELiteFactorizedDecoder


@dataclass(frozen=True)
class CrossingFactorizedDecoderFields:
    """Auditable fields produced by one CR-LVEC forward."""

    baseline_logits: Tensor
    raw_evidence: Tensor
    occupancy_burden: Tensor
    crossing_margin: Tensor
    evidence: Tensor
    logits: Tensor
    projected_occupancy: Tensor
    local_occupancy_count: Tensor
    native_subpixel_size: tuple[int, int]
    output_size: tuple[int, int]
    field_resize_applied: bool


def crossing_recoverable_evidence(crossing_margin: Tensor) -> Tensor:
    """Return positive exponential crossing with its full continuation.

    Let ``u`` denote ``crossing_margin``.  The observable forward value is

    ``f(u) = 0 if u <= 0 else expm1(u)``.

    Backpropagation uses the explicit recovery carrier ``s(u) = exp(u)`` on
    the complete supported numeric axis.  Its derivative is ``exp(u)`` from
    both sides and equals one at the crossing boundary.  The detach
    construction changes no observable forward value and adds no parameter.
    """

    if not isinstance(crossing_margin, Tensor):
        raise TypeError("crossing_margin must be a tensor")
    if not crossing_margin.is_floating_point():
        raise TypeError("crossing_margin must be floating point")
    continuation = torch.expm1(crossing_margin)
    recovery = torch.exp(crossing_margin)
    finite_contract = (
        torch.isfinite(crossing_margin)
        & torch.isfinite(continuation)
        & torch.isfinite(recovery)
        & (recovery > 0.0)
    )
    if not finite_contract.all():
        raise ValueError(
            "crossing margin, continuation, and recovery must remain "
            "finite with nonzero recovery"
        )
    forward_evidence = torch.where(
        crossing_margin <= 0.0,
        torch.zeros_like(continuation),
        continuation,
    )
    return forward_evidence.detach() + (
        recovery - recovery.detach()
    )


def _validate_float_field(
    value: Tensor,
    *,
    name: str,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Validate metadata without synchronizing once per output field."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != shape:
        raise AssertionError(f"{name} has an invalid shape")
    if not value.is_floating_point() or value.dtype != dtype:
        raise TypeError(f"{name} has an invalid dtype")
    if value.device != device:
        raise ValueError(f"{name} is on an invalid device")


class CURELiteCrossingFactorizedDecoder(CURELiteFactorizedDecoder):
    """The unchanged v4 topology with the single CR-LVEC v7 equation."""

    config: CrossingFactorizedDecoderConfig

    def __init__(
        self,
        config: CrossingFactorizedDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        if isinstance(config, CrossingFactorizedDecoderConfig):
            if feature_channels is not None or feature_stride is not None:
                raise ValueError(
                    "do not override an explicit "
                    "CrossingFactorizedDecoderConfig"
                )
            resolved = config
        elif config is None:
            if feature_channels is None or feature_stride is None:
                raise TypeError(
                    "CrossingFactorizedDecoderConfig or "
                    "feature_channels/feature_stride is required"
                )
            resolved = CrossingFactorizedDecoderConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
            )
        else:
            raise TypeError(
                "config must be CrossingFactorizedDecoderConfig or None"
            )

        super().__init__(resolved.to_v4_topology_config())
        self.config = resolved
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != resolved.expected_parameter_count:
            raise AssertionError(
                "CR-LVEC parameter count differs from the frozen topology"
            )

    def burden_field(
        self,
        occupancy: Tensor,
        *,
        feature_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return projected occupancy, 3x3 count, and output-grid burden.

        This is the direct v7 path.  It deliberately does not call the parent
        ``vacancy_field`` because CR-LVEC has no reciprocal-vacancy term.
        """

        if not isinstance(occupancy, Tensor):
            raise TypeError("occupancy must be a tensor")
        if (
            occupancy.ndim != 4
            or occupancy.shape[1] != 1
            or occupancy.shape[0] < 1
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
        output_size = tuple(int(value) for value in occupancy.shape[-2:])
        burden = F.interpolate(
            torch.log1p(count),
            size=output_size,
            mode="nearest",
        )
        return (
            projected.contiguous(),
            count.contiguous(),
            burden.contiguous(),
        )

    def _validate_fields(
        self,
        fields: CrossingFactorizedDecoderFields,
        *,
        feature: Tensor,
        occupancy: Tensor,
    ) -> None:
        batch_size = int(feature.shape[0])
        output_shape = (
            batch_size,
            1,
            int(occupancy.shape[-2]),
            int(occupancy.shape[-1]),
        )
        feature_shape = (
            batch_size,
            1,
            int(feature.shape[-2]),
            int(feature.shape[-1]),
        )
        for name in (
            "baseline_logits",
            "raw_evidence",
            "occupancy_burden",
            "crossing_margin",
            "evidence",
            "logits",
        ):
            _validate_float_field(
                getattr(fields, name),
                name=name,
                shape=output_shape,
                dtype=feature.dtype,
                device=feature.device,
            )

        projected = fields.projected_occupancy
        if not isinstance(projected, Tensor):
            raise TypeError("projected_occupancy must be a tensor")
        if tuple(projected.shape) != feature_shape:
            raise AssertionError(
                "projected_occupancy has an invalid shape"
            )
        if projected.dtype != torch.bool:
            raise TypeError("projected_occupancy must be bool")
        if projected.device != feature.device:
            raise ValueError(
                "projected_occupancy is on an invalid device"
            )

        _validate_float_field(
            fields.local_occupancy_count,
            name="local_occupancy_count",
            shape=feature_shape,
            dtype=feature.dtype,
            device=feature.device,
        )

        native = fields.native_subpixel_size
        output = fields.output_size
        if (
            len(native) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in native
            )
        ):
            raise AssertionError("native_subpixel_size is invalid")
        if output != tuple(int(value) for value in occupancy.shape[-2:]):
            raise AssertionError("output_size is invalid")
        if fields.field_resize_applied != (native != output):
            raise AssertionError("field_resize_applied is invalid")

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CrossingFactorizedDecoderFields:
        """Return all v7 fields without changing the frozen module topology."""

        output_size = self._validate_inputs(feature, occupancy)

        detached = feature.detach()
        trunk0 = F.silu(self.stem_norm(self.stem(detached)))
        residual = self.pointwise(
            F.silu(self.depthwise_norm(self.depthwise(trunk0)))
        )
        trunk = trunk0 + self.config.trunk_residual_scale * residual

        baseline_native = self.pixel_shuffle(self.baseline_head(trunk))
        evidence_raw_native = self.pixel_shuffle(self.evidence_head(trunk))
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
            raw_evidence = F.interpolate(
                evidence_raw_native,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        else:
            baseline_raw = baseline_native
            raw_evidence = evidence_raw_native

        baseline_logits = -F.softplus(
            self.baseline_raw.reshape(1, 1, 1, 1)
            + baseline_raw
        )
        projected, count, occupancy_burden = self.burden_field(
            occupancy,
            feature_size=tuple(int(value) for value in feature.shape[-2:]),
        )
        crossing_margin = raw_evidence - occupancy_burden
        evidence = crossing_recoverable_evidence(crossing_margin)
        logits = baseline_logits + evidence

        fields = CrossingFactorizedDecoderFields(
            baseline_logits=baseline_logits,
            raw_evidence=raw_evidence,
            occupancy_burden=occupancy_burden,
            crossing_margin=crossing_margin,
            evidence=evidence,
            logits=logits,
            projected_occupancy=projected,
            local_occupancy_count=count,
            native_subpixel_size=native_size,
            output_size=output_size,
            field_resize_applied=resize,
        )
        self._validate_fields(
            fields,
            feature=feature,
            occupancy=occupancy,
        )
        return fields


__all__ = [
    "CURELiteCrossingFactorizedDecoder",
    "CrossingFactorizedDecoderFields",
    "crossing_recoverable_evidence",
]
