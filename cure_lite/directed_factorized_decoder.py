"""Directed evidence activation for the additive CURE-Lite D-SVEF v5."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .directed_factorized_config import DirectedFactorizedDecoderConfig
from .factorized_decoder import (
    CURELiteFactorizedDecoder,
    FactorizedDecoderFields,
)


def directed_evidence_activation(raw_evidence: Tensor) -> Tensor:
    """Map signed raw evidence to a one-sided, zero-anchored field.

    The operator is parameter-free and is applied after any raw-field resize:

    ``relu(softplus(raw_evidence) - softplus(0))``.
    """

    if not isinstance(raw_evidence, Tensor):
        raise TypeError("raw_evidence must be a tensor")
    if not raw_evidence.is_floating_point():
        raise TypeError("raw_evidence must be floating point")
    if not torch.isfinite(raw_evidence).all():
        raise ValueError("raw_evidence must be finite")
    zero = raw_evidence.new_zeros(())
    evidence = F.relu(F.softplus(raw_evidence) - F.softplus(zero))
    return torch.where(
        raw_evidence <= 0.0,
        torch.zeros_like(evidence),
        evidence,
    )


class CURELiteDirectedFactorizedDecoder(CURELiteFactorizedDecoder):
    """The v4 topology with one parameter-free directed activation."""

    config: DirectedFactorizedDecoderConfig

    def __init__(
        self,
        config: DirectedFactorizedDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        if isinstance(config, DirectedFactorizedDecoderConfig):
            if feature_channels is not None or feature_stride is not None:
                raise ValueError(
                    "do not override an explicit "
                    "DirectedFactorizedDecoderConfig"
                )
            resolved = config
        elif config is None:
            if feature_channels is None or feature_stride is None:
                raise TypeError(
                    "DirectedFactorizedDecoderConfig or "
                    "feature_channels/feature_stride is required"
                )
            resolved = DirectedFactorizedDecoderConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
            )
        else:
            raise TypeError(
                "config must be DirectedFactorizedDecoderConfig or None"
            )

        super().__init__(resolved.to_v4_topology_config())
        self.config = resolved
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != resolved.expected_parameter_count:
            raise AssertionError(
                "D-SVEF parameter count differs from the frozen topology"
            )

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> FactorizedDecoderFields:
        """Return v5 fields; only the evidence activation differs from v4."""

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
        evidence = directed_evidence_activation(evidence_raw)
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
            raise AssertionError(
                "D-SVEF fields must share the evaluation shape"
            )
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


__all__ = [
    "CURELiteDirectedFactorizedDecoder",
    "directed_evidence_activation",
]
