"""Polarity-recoverable evidence for additive CURE-Lite PR-SVEF v6."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .factorized_decoder import (
    CURELiteFactorizedDecoder,
    FactorizedDecoderFields,
)
from .recoverable_factorized_config import (
    RecoverableFactorizedDecoderConfig,
)


def polarity_recoverable_evidence(raw_evidence: Tensor) -> Tensor:
    """Apply the frozen PR-SVEF forward value and surrogate gradient.

    Forward:
        ``softplus(relu(raw)^2) - softplus(0)``, with an explicit exact-zero
        branch for ``raw <= 0``.

    Backward:
        the positive branch keeps the true v4 derivative, while ``raw <= 0``
        uses the derivative of centered softplus.  The detach construction
        changes no forward value and introduces no trainable parameter.
    """

    if not isinstance(raw_evidence, Tensor):
        raise TypeError("raw_evidence must be a tensor")
    if not raw_evidence.is_floating_point():
        raise TypeError("raw_evidence must be floating point")
    if not torch.isfinite(raw_evidence).all():
        raise ValueError("raw_evidence must be finite")

    zero = raw_evidence.new_zeros(())
    positive = F.relu(raw_evidence)
    squared = positive.square()
    forward_evidence = F.softplus(squared) - F.softplus(zero)
    forward_evidence = torch.where(
        raw_evidence <= 0.0,
        torch.zeros_like(forward_evidence),
        forward_evidence,
    )

    recovery = F.softplus(raw_evidence) - F.softplus(zero)
    surrogate = torch.where(
        raw_evidence > 0.0,
        forward_evidence,
        recovery,
    )
    return (
        forward_evidence.detach()
        + surrogate
        - surrogate.detach()
    )


class CURELiteRecoverableFactorizedDecoder(CURELiteFactorizedDecoder):
    """The unchanged v4 topology with one PR-SVEF evidence operator."""

    config: RecoverableFactorizedDecoderConfig

    def __init__(
        self,
        config: RecoverableFactorizedDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        if isinstance(config, RecoverableFactorizedDecoderConfig):
            if feature_channels is not None or feature_stride is not None:
                raise ValueError(
                    "do not override an explicit "
                    "RecoverableFactorizedDecoderConfig"
                )
            resolved = config
        elif config is None:
            if feature_channels is None or feature_stride is None:
                raise TypeError(
                    "RecoverableFactorizedDecoderConfig or "
                    "feature_channels/feature_stride is required"
                )
            resolved = RecoverableFactorizedDecoderConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
            )
        else:
            raise TypeError(
                "config must be RecoverableFactorizedDecoderConfig or None"
            )

        super().__init__(resolved.to_v4_topology_config())
        self.config = resolved
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != resolved.expected_parameter_count:
            raise AssertionError(
                "PR-SVEF parameter count differs from the frozen topology"
            )

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> FactorizedDecoderFields:
        """Return v6 fields; only the evidence operator differs from v4."""

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
        evidence = polarity_recoverable_evidence(evidence_raw)
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
                "PR-SVEF fields must share the evaluation shape"
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
    "CURELiteRecoverableFactorizedDecoder",
    "polarity_recoverable_evidence",
]
