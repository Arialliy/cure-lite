"""Coverage-Conserving Subpixel Evidence Allocation for CURE-Lite v8.

CC-SEA replaces v7's independent per-pixel crossing with one local state
equation.  Each feature cell produces one coverage-conditioned evidence
budget, and its ``stride**2`` output phases compete for that same budget.
The shared trunk, heads, parameter count, loss, and inference graph remain
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import functional as F

from .conservative_factorized_config import (
    ConservativeFactorizedDecoderConfig,
)
from .crossing_factorized_decoder import crossing_recoverable_evidence
from .decoder import project_occupancy_to_feature_grid
from .factorized_decoder import CURELiteFactorizedDecoder


@dataclass(frozen=True)
class ConservativeFactorizedDecoderFields:
    """Auditable fields produced by one CC-SEA forward."""

    baseline_logits: Tensor
    raw_phase_evidence: Tensor
    common_mode_phase_evidence: Tensor
    occupancy_burden: Tensor
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


def coverage_conserving_phase_evidence(
    raw_phase_evidence: Tensor,
    occupancy_burden: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Allocate one coverage-conditioned budget over subpixel phases.

    For ``P`` phase logits ``r_j`` in one feature cell,

    ``mu = mean_j(r_j)``,
    ``u = mu - b``,
    ``M = crossing_recoverable_evidence(u)``,
    ``alpha_j = softmax_j(r_j)``,
    ``e_j = M * alpha_j``.

    Hence ``e_j >= 0`` and ``sum_j(e_j) = M`` up to floating-point
    reduction error.  The arithmetic common mode and the zero-mean phase
    contrasts are orthogonal coordinates of the same head output: a common
    shift changes only the budget, while a zero-mean contrast changes only
    the allocation.  Occupancy changes the shared budget but cannot
    independently amplify every phase.
    """

    if not isinstance(raw_phase_evidence, Tensor):
        raise TypeError("raw_phase_evidence must be a tensor")
    if not isinstance(occupancy_burden, Tensor):
        raise TypeError("occupancy_burden must be a tensor")
    if (
        raw_phase_evidence.ndim != 4
        or raw_phase_evidence.shape[0] < 1
        or raw_phase_evidence.shape[1] < 1
        or min(raw_phase_evidence.shape[-2:]) < 1
    ):
        raise ValueError(
            "raw_phase_evidence must have shape [B,P,h,w]"
        )
    expected_burden_shape = (
        int(raw_phase_evidence.shape[0]),
        1,
        int(raw_phase_evidence.shape[2]),
        int(raw_phase_evidence.shape[3]),
    )
    if tuple(occupancy_burden.shape) != expected_burden_shape:
        raise ValueError(
            "occupancy_burden must have shape [B,1,h,w]"
        )
    if (
        not raw_phase_evidence.is_floating_point()
        or not occupancy_burden.is_floating_point()
    ):
        raise TypeError("CC-SEA fields must be floating point")
    if raw_phase_evidence.dtype != occupancy_burden.dtype:
        raise TypeError("CC-SEA fields must share a dtype")
    if raw_phase_evidence.device != occupancy_burden.device:
        raise ValueError("CC-SEA fields must share a device")
    if not torch.isfinite(raw_phase_evidence).all():
        raise ValueError("raw_phase_evidence must be finite")
    if not torch.isfinite(occupancy_burden).all():
        raise ValueError("occupancy_burden must be finite")
    if torch.any(occupancy_burden < 0.0):
        raise ValueError("occupancy_burden must be nonnegative")

    common_mode = raw_phase_evidence.mean(dim=1, keepdim=True)
    phase_contrast = raw_phase_evidence - common_mode
    budget_margin = common_mode - occupancy_burden
    evidence_budget = crossing_recoverable_evidence(budget_margin)
    phase_allocation = torch.softmax(
        phase_contrast,
        dim=1,
    )
    allocated = phase_allocation * evidence_budget

    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            common_mode,
            budget_margin,
            evidence_budget,
            phase_allocation,
            allocated,
        )
    ):
        raise ValueError("CC-SEA operator produced a nonfinite field")
    return (
        common_mode,
        budget_margin,
        evidence_budget,
        phase_allocation,
        allocated,
    )


class CURELiteConservativeFactorizedDecoder(
    CURELiteFactorizedDecoder
):
    """The unchanged v4 topology with the single CC-SEA v8 equation."""

    config: ConservativeFactorizedDecoderConfig

    def __init__(
        self,
        config: ConservativeFactorizedDecoderConfig | None = None,
        *,
        feature_channels: int | None = None,
        feature_stride: int | None = None,
    ) -> None:
        if isinstance(config, ConservativeFactorizedDecoderConfig):
            if feature_channels is not None or feature_stride is not None:
                raise ValueError(
                    "do not override an explicit "
                    "ConservativeFactorizedDecoderConfig"
                )
            resolved = config
        elif config is None:
            if feature_channels is None or feature_stride is None:
                raise TypeError(
                    "ConservativeFactorizedDecoderConfig or "
                    "feature_channels/feature_stride is required"
                )
            resolved = ConservativeFactorizedDecoderConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
            )
        else:
            raise TypeError(
                "config must be "
                "ConservativeFactorizedDecoderConfig or None"
            )

        super().__init__(resolved.to_v4_topology_config())
        self.config = resolved
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != resolved.expected_parameter_count:
            raise AssertionError(
                "CC-SEA parameter count differs from the frozen topology"
            )

    def native_burden_field(
        self,
        occupancy: Tensor,
        *,
        feature_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return projected occupancy, local count, and native burden."""

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
        burden = torch.log1p(count)
        return (
            projected.contiguous(),
            count.contiguous(),
            burden.contiguous(),
        )

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> ConservativeFactorizedDecoderFields:
        """Return all v8 fields without changing the module topology."""

        output_size = self._validate_inputs(feature, occupancy)
        detached = feature.detach()
        trunk0 = F.silu(self.stem_norm(self.stem(detached)))
        residual = self.pointwise(
            F.silu(self.depthwise_norm(self.depthwise(trunk0)))
        )
        trunk = trunk0 + self.config.trunk_residual_scale * residual

        baseline_phase = self.baseline_head(trunk)
        raw_phase_evidence = self.evidence_head(trunk)
        projected, count, burden = self.native_burden_field(
            occupancy,
            feature_size=tuple(
                int(value) for value in feature.shape[-2:]
            ),
        )
        (
            common_mode,
            budget_margin,
            evidence_budget,
            phase_allocation,
            allocated_phase_evidence,
        ) = coverage_conserving_phase_evidence(
            raw_phase_evidence,
            burden,
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

        return ConservativeFactorizedDecoderFields(
            baseline_logits=baseline_logits,
            raw_phase_evidence=raw_phase_evidence,
            common_mode_phase_evidence=common_mode,
            occupancy_burden=burden,
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
    "CURELiteConservativeFactorizedDecoder",
    "ConservativeFactorizedDecoderFields",
    "coverage_conserving_phase_evidence",
]
