"""Subpixel Vacancy-Evidence Factorization for CURE-Lite v4.

This module is additive.  It does not alter the frozen v1 decoder or the
v3 OC-APTO objective.  The trainable path consumes detached detector features
only; occupancy participates through one fixed inverse-count vacancy field.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import expm1, log

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoder import project_occupancy_to_feature_grid
from .factorized_config import FactorizedDecoderConfig


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse softplus requires a positive value")
    return value + log(-expm1(-value))


@dataclass(frozen=True)
class FactorizedDecoderFields:
    """Auditable fields produced by one SVEF forward."""

    baseline_logits: Tensor
    evidence: Tensor
    vacancy: Tensor
    logits: Tensor
    projected_occupancy: Tensor
    local_occupancy_count: Tensor
    native_subpixel_size: tuple[int, int]
    output_size: tuple[int, int]
    field_resize_applied: bool


class CURELiteFactorizedDecoder(nn.Module):
    """One shared high-resolution decoder with fixed vacancy modulation."""

    def __init__(
        self,
        config: FactorizedDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(config, FactorizedDecoderConfig):
            if feature_channels is not None or feature_stride is not None:
                raise ValueError(
                    "do not override an explicit FactorizedDecoderConfig"
                )
            resolved = config
        elif config is None:
            if feature_channels is None or feature_stride is None:
                raise TypeError(
                    "FactorizedDecoderConfig or feature_channels/feature_stride "
                    "is required"
                )
            resolved = FactorizedDecoderConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
            )
        else:
            raise TypeError("config must be FactorizedDecoderConfig or None")

        self.config = resolved
        self.feature_channels = resolved.feature_channels
        self.feature_stride = resolved.feature_stride
        self.phase_channels = resolved.phase_channels

        self.stem = nn.Conv2d(
            resolved.feature_channels,
            resolved.width,
            kernel_size=1,
            bias=False,
        )
        self.stem_norm = nn.GroupNorm(
            resolved.groups,
            resolved.width,
            affine=False,
        )
        self.depthwise = nn.Conv2d(
            resolved.width,
            resolved.width,
            kernel_size=3,
            padding=1,
            groups=resolved.width,
            bias=False,
        )
        self.depthwise_norm = nn.GroupNorm(
            resolved.groups,
            resolved.width,
            affine=False,
        )
        self.pointwise = nn.Conv2d(
            resolved.width,
            resolved.width,
            kernel_size=1,
            bias=False,
        )
        self.baseline_head = nn.Conv2d(
            resolved.width,
            resolved.phase_channels,
            kernel_size=1,
            bias=False,
        )
        self.evidence_head = nn.Conv2d(
            resolved.width,
            resolved.phase_channels,
            kernel_size=1,
            bias=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(resolved.feature_stride)

        baseline_logit = log(
            resolved.baseline_probability
            / (1.0 - resolved.baseline_probability)
        )
        self.baseline_raw = nn.Parameter(
            torch.tensor(
                _inverse_softplus(-baseline_logit),
                dtype=torch.float32,
            )
        )
        self.register_buffer(
            "_vacancy_kernel",
            torch.ones(
                1,
                1,
                resolved.vacancy_kernel_size,
                resolved.vacancy_kernel_size,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self._reset_parameters()

        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != resolved.expected_parameter_count:
            raise AssertionError(
                "SVEF parameter count differs from its frozen topology"
            )

    def _reset_parameters(self) -> None:
        for layer in (self.stem, self.depthwise, self.pointwise):
            nn.init.kaiming_normal_(
                layer.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
        nn.init.zeros_(self.baseline_head.weight)
        nn.init.xavier_normal_(self.evidence_head.weight, gain=0.25)

    def _validate_inputs(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> tuple[int, int]:
        if not isinstance(feature, Tensor) or not isinstance(occupancy, Tensor):
            raise TypeError("feature and occupancy must be tensors")
        if feature.ndim != 4:
            raise ValueError("feature must have shape [B,C,h,w]")
        if occupancy.ndim != 4 or occupancy.shape[1] != 1:
            raise ValueError("occupancy must have shape [B,1,H,W]")
        if feature.shape[0] < 1 or feature.shape[0] != occupancy.shape[0]:
            raise ValueError(
                "feature and occupancy batches must agree and be non-empty"
            )
        if feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"expected {self.feature_channels} feature channels, "
                f"got {feature.shape[1]}"
            )
        if any(size < 1 for size in feature.shape[-2:]):
            raise ValueError("feature spatial dimensions must be positive")
        if any(size < 1 for size in occupancy.shape[-2:]):
            raise ValueError("occupancy spatial dimensions must be positive")
        if not feature.is_floating_point():
            raise TypeError("feature must be floating point")
        if feature.dtype != self.stem.weight.dtype:
            raise TypeError(
                "feature dtype must match decoder weights "
                f"({feature.dtype} != {self.stem.weight.dtype})"
            )
        if occupancy.dtype != torch.bool:
            raise TypeError("occupancy must be bool")
        if feature.device != occupancy.device:
            raise ValueError("feature and occupancy must share a device")
        if feature.device != self.stem.weight.device:
            raise ValueError("inputs and decoder parameters must share a device")
        if any(
            feature_size > output_size
            for feature_size, output_size in zip(
                feature.shape[-2:],
                occupancy.shape[-2:],
                strict=True,
            )
        ):
            raise ValueError("occupancy projection may not upsample")
        return tuple(int(value) for value in occupancy.shape[-2:])

    def vacancy_field(
        self,
        occupancy: Tensor,
        *,
        feature_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return projected occupancy, local count, and final vacancy field."""

        if not isinstance(occupancy, Tensor):
            raise TypeError("occupancy must be a tensor")
        if occupancy.ndim != 4 or occupancy.shape[1] != 1:
            raise ValueError("occupancy must have shape [B,1,H,W]")
        if occupancy.shape[0] < 1:
            raise ValueError("occupancy batch must be non-empty")
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
            raise ValueError("feature_size must contain two positive integers")
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
        vacancy_small = count.add(1.0).reciprocal()
        output_size = tuple(int(value) for value in occupancy.shape[-2:])
        vacancy = F.interpolate(
            vacancy_small,
            size=output_size,
            mode="nearest",
        )
        return (
            projected.contiguous(),
            count.contiguous(),
            vacancy.contiguous(),
        )

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> FactorizedDecoderFields:
        """Return the complete factorization without changing forward inputs."""

        output_size = self._validate_inputs(feature, occupancy)
        detached = feature.detach()
        trunk0 = F.silu(self.stem_norm(self.stem(detached)))
        residual = self.pointwise(
            F.silu(self.depthwise_norm(self.depthwise(trunk0)))
        )
        trunk = (
            trunk0
            + self.config.trunk_residual_scale * residual
        )

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
            evidence_raw = F.interpolate(
                evidence_raw_native,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        else:
            baseline_raw = baseline_native
            evidence_raw = evidence_raw_native

        baseline_logits = -F.softplus(
            self.baseline_raw.reshape(1, 1, 1, 1)
            + baseline_raw
        )
        evidence_squared = evidence_raw.square()
        softplus_zero = F.softplus(evidence_squared.new_zeros(()))
        evidence = (
            F.softplus(evidence_squared)
            - softplus_zero
        )
        evidence = torch.where(
            evidence_squared == 0.0,
            torch.zeros_like(evidence),
            evidence,
        )
        projected, count, vacancy = self.vacancy_field(
            occupancy,
            feature_size=tuple(int(value) for value in feature.shape[-2:]),
        )
        logits = baseline_logits + evidence * vacancy
        if (
            baseline_logits.shape != vacancy.shape
            or evidence.shape != vacancy.shape
            or logits.shape != vacancy.shape
        ):
            raise AssertionError("SVEF fields must share the evaluation shape")
        return FactorizedDecoderFields(
            baseline_logits=baseline_logits,
            evidence=evidence,
            vacancy=vacancy,
            logits=logits,
            projected_occupancy=projected,
            local_occupancy_count=count,
            native_subpixel_size=native_size,
            output_size=output_size,
            field_resize_applied=resize,
        )

    def forward(self, feature: Tensor, occupancy: Tensor) -> Tensor:
        return self.forward_fields(feature, occupancy).logits


__all__ = [
    "CURELiteFactorizedDecoder",
    "FactorizedDecoderFields",
]
