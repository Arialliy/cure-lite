"""Phase-preserving coverage-state level-set field.

The legacy CSLF compresses each output-stride occupancy cell to one Boolean
value.  PPCE keeps the same single-path field architecture while replacing
that lossy state coordinate with the exact PixelUnshuffle phase basis:

``[B,1,H,W] -> [B,s**2,H/s,W/s]``.

The phase order is the row-major order used by :class:`torch.nn.PixelShuffle`,
so the representation is exactly invertible and aligned with the output field
phases.  No prediction head, auxiliary branch, or post-processing operation is
introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
    normalize_cslf_feature,
)


CSLF_PHASE_PRESERVING_COVERAGE_POLICY = (
    "lossless_bool_pixel_unshuffle_row_major_phase_coverage_v1"
)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def pixel_unshuffle_bool_occupancy(
    occupancy: Tensor,
    *,
    stride: int,
) -> Tensor:
    """Return the lossless row-major phase basis of a Boolean occupancy.

    Conversion through FP32 is exact for Boolean values and permits use of
    PyTorch's canonical PixelUnshuffle/PixelShuffle pair on every supported
    execution device.  The roundtrip assertion binds the phase convention
    instead of relying on a separately reimplemented index formula.
    """

    if (
        not isinstance(occupancy, Tensor)
        or occupancy.dtype != torch.bool
        or occupancy.ndim != 4
        or occupancy.shape[0] < 1
        or occupancy.shape[1] != 1
        or min(occupancy.shape[-2:]) < 1
    ):
        raise TypeError("occupancy must be bool [B,1,H,W]")
    stride = _positive_int(stride, name="stride")
    height, width = (int(value) for value in occupancy.shape[-2:])
    if height % stride != 0 or width % stride != 0:
        raise ValueError("occupancy grid must be divisible by stride")
    phase = F.pixel_unshuffle(
        occupancy.to(dtype=torch.float32),
        stride,
    ).to(dtype=torch.bool).contiguous()
    reconstructed = F.pixel_shuffle(
        phase.to(dtype=torch.float32),
        stride,
    ).to(dtype=torch.bool)
    if not torch.equal(reconstructed, occupancy):
        raise AssertionError(
            "phase occupancy roundtrip changed the full-grid state"
        )
    return phase


@dataclass(frozen=True)
class CoverageStatePhasePreservingConfig(CoverageStateLevelSetConfig):
    """Fixed PPCE structure compatible with the legacy CSLF config."""

    coverage_policy: str = CSLF_PHASE_PRESERVING_COVERAGE_POLICY

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.coverage_policy != CSLF_PHASE_PRESERVING_COVERAGE_POLICY:
            raise ValueError("PPCE fixes coverage_policy")

    @property
    def phase_occupancy_channels(self) -> int:
        return self.feature_stride**2

    @property
    def expected_parameter_count(self) -> int:
        input_projection = (
            (
                self.feature_channels
                + self.phase_occupancy_channels
            )
            * self.width
            * 3
            * 3
        )
        spatial_mixing = self.width * 3 * 3
        phase_projection = (
            self.width * self.phase_channels + self.phase_channels
        )
        return (
            input_projection
            + spatial_mixing
            + phase_projection
        )


@dataclass(frozen=True)
class CoverageStatePhasePreservingFields:
    """Auditable tensors from one phase-preserving CSLF forward."""

    encoded_feature: Tensor
    phase_occupancy: Tensor
    hidden: Tensor
    native_phase_field: Tensor
    field: Tensor
    output_size: tuple[int, int]


class CURELitePhasePreservingCoverageStateLevelSet(
    CURELiteCoverageStateLevelSet
):
    """One CSLF path conditioned on exact output-grid occupancy phases."""

    config: CoverageStatePhasePreservingConfig

    def __init__(
        self,
        config: CoverageStatePhasePreservingConfig,
    ) -> None:
        if not isinstance(config, CoverageStatePhasePreservingConfig):
            raise TypeError(
                "config must be CoverageStatePhasePreservingConfig"
            )
        nn.Module.__init__(self)
        self.config = config
        self.input_projection = nn.Conv2d(
            (
                config.feature_channels
                + config.phase_occupancy_channels
            ),
            config.width,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.spatial_mixing = nn.Conv2d(
            config.width,
            config.width,
            kernel_size=3,
            padding=1,
            groups=config.width,
            bias=False,
        )
        self.phase_projection = nn.Conv2d(
            config.width,
            config.phase_channels,
            kernel_size=1,
            bias=True,
        )
        self.pixel_shuffle = nn.PixelShuffle(config.feature_stride)
        self._reset_parameters()
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != config.expected_parameter_count:
            raise AssertionError(
                "PPCE parameter count differs from its contract"
            )

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CoverageStatePhasePreservingFields:
        """Return the single-path PPCE level-set state equation."""

        output_size = self._validate_inputs(feature, occupancy)
        encoded_feature = normalize_cslf_feature(
            feature,
            epsilon=self.config.normalization_epsilon,
        )
        phase_occupancy = pixel_unshuffle_bool_occupancy(
            occupancy,
            stride=self.config.feature_stride,
        )
        if tuple(phase_occupancy.shape[-2:]) != tuple(
            encoded_feature.shape[-2:]
        ):
            raise AssertionError(
                "phase occupancy does not align with the feature grid"
            )
        state = torch.cat(
            (
                encoded_feature,
                phase_occupancy.to(dtype=encoded_feature.dtype),
            ),
            dim=1,
        )
        hidden = F.silu(self.input_projection(state))
        hidden = hidden + F.silu(self.spatial_mixing(hidden))
        native_phase = self.phase_projection(hidden)
        field = self.pixel_shuffle(native_phase)
        fields = CoverageStatePhasePreservingFields(
            encoded_feature=encoded_feature.contiguous(),
            phase_occupancy=phase_occupancy.contiguous(),
            hidden=hidden.contiguous(),
            native_phase_field=native_phase.contiguous(),
            field=field.contiguous(),
            output_size=output_size,
        )
        self._validate_phase_preserving_fields(
            fields,
            feature=feature,
            occupancy=occupancy,
        )
        return fields

    def _validate_phase_preserving_fields(
        self,
        fields: CoverageStatePhasePreservingFields,
        *,
        feature: Tensor,
        occupancy: Tensor,
    ) -> None:
        batch, _, height, width = feature.shape
        expected = (
            (
                "encoded_feature",
                fields.encoded_feature,
                tuple(feature.shape),
            ),
            (
                "phase_occupancy",
                fields.phase_occupancy,
                (
                    batch,
                    self.config.phase_occupancy_channels,
                    height,
                    width,
                ),
            ),
            (
                "hidden",
                fields.hidden,
                (batch, self.config.width, height, width),
            ),
            (
                "native_phase_field",
                fields.native_phase_field,
                (
                    batch,
                    self.config.phase_channels,
                    height,
                    width,
                ),
            ),
            ("field", fields.field, tuple(occupancy.shape)),
        )
        for name, value, shape in expected:
            if tuple(value.shape) != shape:
                raise AssertionError(f"{name} has an invalid shape")
            if value.device != feature.device:
                raise ValueError(f"{name} device differs from feature")
            if name == "phase_occupancy":
                if value.dtype != torch.bool:
                    raise TypeError("phase_occupancy must be bool")
            elif value.dtype != feature.dtype:
                raise TypeError(f"{name} dtype differs from feature")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"{name} must be finite")


def build_coverage_state_level_set(
    config: CoverageStateLevelSetConfig,
) -> CURELiteCoverageStateLevelSet:
    """Build the exact model family named by one structural config."""

    from .coverage_state_centered_mixed_interaction import (
        CURELiteCenteredMixedInteractionLevelSet,
        CoverageStateCenteredMixedInteractionConfig,
    )

    if isinstance(config, CoverageStateCenteredMixedInteractionConfig):
        return CURELiteCenteredMixedInteractionLevelSet(config)
    if isinstance(config, CoverageStatePhasePreservingConfig):
        return CURELitePhasePreservingCoverageStateLevelSet(config)
    if isinstance(config, CoverageStateLevelSetConfig):
        return CURELiteCoverageStateLevelSet(config)
    raise TypeError(
        "config must be a registered coverage-state level-set config"
    )


__all__ = [
    "CSLF_PHASE_PRESERVING_COVERAGE_POLICY",
    "CURELitePhasePreservingCoverageStateLevelSet",
    "CoverageStatePhasePreservingConfig",
    "CoverageStatePhasePreservingFields",
    "build_coverage_state_level_set",
    "pixel_unshuffle_bool_occupancy",
]
