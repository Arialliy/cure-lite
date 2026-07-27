"""Coverage-state level-set field for the next CURE-Lite candidate.

The model predicts one non-saturating scalar field from the detector-independent
``(F_b, O)`` contract.  The zero level set is the residual target boundary:

``phi < 0``
    residual target support;

``phi >= 0``
    no residual target support.

Base occupancy is both an input state and a hard, non-writable output set.
The module deliberately has no object proposal head, null head, component
tree, transport solver, or post-processing branch.  Empty residual states,
multiple residual components, and their spatial support share the same scalar
field representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoder import project_occupancy_to_feature_grid
CSLF_FIELD_POLICY = (
    "single_non_saturating_phase_resolved_level_set_field_v2"
)
CSLF_TARGET_POLICY = (
    "fixed_amplitude_truncated_signed_chessboard_distance_on_masked_grid_v3"
)
CSLF_OUTPUT_POLICY = (
    "negative_zero_level_set_then_occupancy_exclusion_and_hard_union_v1"
)
CSLF_FEATURE_POLICY = (
    "samplewise_global_rms_normalized_relative_spatial_amplitude_v2"
)
CSLF_FIELD_AMPLITUDE = 0.9
CSLF_INITIAL_FIELD_VALUE = 0.9
CSLF_NORMALIZATION_EPSILON = 1.0e-6
CSLF_NUMERICAL_POLICY = "float32_field_and_geometry_without_amp_v1"


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class CoverageStateLevelSetConfig:
    """Structural contract for the detector-independent level-set field."""

    feature_channels: int
    feature_stride: int
    width: int = 32
    normalization_epsilon: float = CSLF_NORMALIZATION_EPSILON
    field_amplitude: float = CSLF_FIELD_AMPLITUDE
    initial_field_value: float = CSLF_INITIAL_FIELD_VALUE
    field_policy: str = CSLF_FIELD_POLICY
    target_policy: str = CSLF_TARGET_POLICY
    output_policy: str = CSLF_OUTPUT_POLICY
    feature_policy: str = CSLF_FEATURE_POLICY
    numerical_policy: str = CSLF_NUMERICAL_POLICY

    def __post_init__(self) -> None:
        for name in (
            "feature_channels",
            "feature_stride",
            "width",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        if (
            isinstance(self.normalization_epsilon, bool)
            or not isinstance(self.normalization_epsilon, float)
            or self.normalization_epsilon != CSLF_NORMALIZATION_EPSILON
        ):
            raise ValueError("CURE-Lite CSLF fixes normalization_epsilon")
        if (
            isinstance(self.field_amplitude, bool)
            or not isinstance(self.field_amplitude, float)
            or self.field_amplitude != CSLF_FIELD_AMPLITUDE
        ):
            raise ValueError("CURE-Lite CSLF fixes field_amplitude")
        if (
            isinstance(self.initial_field_value, bool)
            or not isinstance(self.initial_field_value, float)
            or self.initial_field_value != CSLF_INITIAL_FIELD_VALUE
        ):
            raise ValueError("CURE-Lite CSLF fixes initial_field_value")
        if self.initial_field_value != self.field_amplitude:
            raise ValueError(
                "initial_field_value must equal the target-field amplitude"
            )
        frozen = {
            "field_policy": CSLF_FIELD_POLICY,
            "target_policy": CSLF_TARGET_POLICY,
            "output_policy": CSLF_OUTPUT_POLICY,
            "feature_policy": CSLF_FEATURE_POLICY,
            "numerical_policy": CSLF_NUMERICAL_POLICY,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"CURE-Lite CSLF fixes {name}")

    @property
    def phase_channels(self) -> int:
        return self.feature_stride**2

    @property
    def truncation_radius(self) -> int:
        """Use the detector stride as the fixed native-grid field radius."""

        return self.feature_stride

    @property
    def expected_parameter_count(self) -> int:
        input_projection = (
            (self.feature_channels + 1) * self.width * 3 * 3
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
class CoverageStateLevelSetFields:
    """Auditable tensors from one CSLF forward."""

    encoded_feature: Tensor
    projected_occupancy: Tensor
    hidden: Tensor
    native_phase_field: Tensor
    field: Tensor
    output_size: tuple[int, int]


def _validate_binary_grid(
    value: Tensor,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> None:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.bool
        or value.ndim != 4
        or value.shape[0] < 1
        or value.shape[1] != 1
        or min(value.shape[-2:]) < 1
    ):
        raise ValueError(f"{name} must be bool [B,1,H,W]")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} shape does not match the reference grid")


def _truncated_chessboard_distance(
    seed: Tensor,
    *,
    radius: int,
    allowed_mask: Tensor,
) -> Tensor:
    """Return distance to ``seed`` truncated to ``radius``.

    A 3 x 3 max-pool realizes one step on the 8-neighbour output grid.  The
    routine is exact for the truncated chessboard metric, device-preserving,
    deterministic, and requires no optional image-processing dependency.
    """

    _validate_binary_grid(seed, name="seed")
    _validate_binary_grid(
        allowed_mask,
        name="allowed_mask",
        shape=tuple(seed.shape),
    )
    if seed.device != allowed_mask.device:
        raise ValueError("seed and allowed_mask must share a device")
    if bool(torch.any(seed & ~allowed_mask)):
        raise ValueError("seed extends outside allowed_mask")
    radius = _positive_int(radius, name="radius")
    distance = torch.full(
        seed.shape,
        radius,
        dtype=torch.int64,
        device=seed.device,
    )
    distance = distance.masked_fill(seed, 0)
    reached = seed & allowed_mask
    for step in range(1, radius + 1):
        expanded = F.max_pool2d(
            reached.to(dtype=torch.float32),
            kernel_size=3,
            stride=1,
            padding=1,
        ).to(dtype=torch.bool) & allowed_mask
        newly_reached = expanded & ~reached
        distance = distance.masked_fill(newly_reached, step)
        reached = expanded
    return distance


def truncated_signed_distance_field(
    target: Tensor,
    valid_mask: Tensor,
    *,
    radius: int,
    amplitude: float = CSLF_FIELD_AMPLITUDE,
) -> Tensor:
    """Convert a binary residual mask into a bounded signed level-set target.

    The returned target lies in ``[-a, a]`` for the fixed target amplitude
    ``a``.  Target pixels are strictly
    negative, valid non-target pixels are strictly positive, and an empty
    target is represented by the all-``+a`` field.  Invalid pixels are set
    to ``+a`` and must remain excluded by the caller's loss measure.
    """

    _validate_binary_grid(target, name="target")
    _validate_binary_grid(
        valid_mask,
        name="valid_mask",
        shape=tuple(target.shape),
    )
    if target.device != valid_mask.device:
        raise ValueError("target and valid_mask must share a device")
    if bool(torch.any(target & ~valid_mask)):
        raise ValueError("target extends outside valid_mask")
    radius = _positive_int(radius, name="radius")
    if (
        isinstance(amplitude, bool)
        or not isinstance(amplitude, float)
        or amplitude != CSLF_FIELD_AMPLITUDE
    ):
        raise ValueError("CURE-Lite CSLF fixes the target-field amplitude")

    target = target & valid_mask
    distance_to_target = _truncated_chessboard_distance(
        target,
        radius=radius,
        allowed_mask=valid_mask,
    )
    background_seed = valid_mask & ~target
    distance_to_background = _truncated_chessboard_distance(
        background_seed,
        radius=radius,
        allowed_mask=valid_mask,
    )
    scale = float(radius)
    outside = (
        distance_to_target.clamp(min=1, max=radius).to(torch.float32)
        / scale * amplitude
    )
    inside = -(
        distance_to_background.clamp(min=1, max=radius).to(torch.float32)
        / scale * amplitude
    )
    field = torch.where(target, inside, outside)
    field = torch.where(
        valid_mask,
        field,
        torch.full_like(field, amplitude),
    )
    if (
        not bool(torch.isfinite(field).all())
        or not bool(
            ((field >= -amplitude) & (field <= amplitude)).all()
        )
        or bool(torch.any(field[target] >= 0.0))
        or bool(torch.any(field[valid_mask & ~target] <= 0.0))
    ):
        raise AssertionError("signed-distance target contract changed")
    return field.contiguous()


class CURELiteCoverageStateLevelSet(nn.Module):
    """One scalar completion field over frozen feature and coverage state."""

    config: CoverageStateLevelSetConfig

    def __init__(self, config: CoverageStateLevelSetConfig) -> None:
        super().__init__()
        if not isinstance(config, CoverageStateLevelSetConfig):
            raise TypeError("config must be CoverageStateLevelSetConfig")
        self.config = config
        self.input_projection = nn.Conv2d(
            config.feature_channels + 1,
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
                "CSLF parameter count differs from its contract"
            )

    @property
    def feature_channels(self) -> int:
        return self.config.feature_channels

    @property
    def feature_stride(self) -> int:
        return self.config.feature_stride

    def _reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.input_projection.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.kaiming_normal_(
            self.spatial_mixing.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.phase_projection.weight)
        nn.init.constant_(
            self.phase_projection.bias,
            self.config.initial_field_value,
        )

    def _validate_inputs(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> tuple[int, int]:
        if (
            not isinstance(feature, Tensor)
            or feature.ndim != 4
            or feature.shape[0] < 1
            or feature.shape[1] != self.config.feature_channels
            or min(feature.shape[-2:]) < 1
            or not feature.is_floating_point()
        ):
            raise ValueError(
                "feature must be floating [B,C,h,w] with configured C"
            )
        _validate_binary_grid(occupancy, name="occupancy")
        if feature.shape[0] != occupancy.shape[0]:
            raise ValueError("feature and occupancy batch sizes differ")
        if feature.device != occupancy.device:
            raise ValueError("feature and occupancy must share a device")
        if feature.device != self.input_projection.weight.device:
            raise ValueError("model and inputs must share a device")
        if (
            feature.dtype != torch.float32
            or self.input_projection.weight.dtype != torch.float32
        ):
            raise TypeError("CSLF fixes model and feature dtype to float32")
        if not bool(torch.isfinite(feature).all()):
            raise ValueError("feature must be finite")
        expected = (
            int(feature.shape[-2]) * self.config.feature_stride,
            int(feature.shape[-1]) * self.config.feature_stride,
        )
        if tuple(occupancy.shape[-2:]) != expected:
            raise ValueError(
                "occupancy size must equal feature size times feature_stride"
            )
        return expected

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CoverageStateLevelSetFields:
        """Return the complete level-set state equation."""

        output_size = self._validate_inputs(feature, occupancy)
        frozen_feature = feature.detach()
        sample_rms = frozen_feature.square().mean(
            dim=(1, 2, 3),
            keepdim=True,
        ).sqrt()
        encoded_feature = frozen_feature / sample_rms.clamp_min(
            self.config.normalization_epsilon
        )
        projected_occupancy = project_occupancy_to_feature_grid(
            occupancy,
            tuple(int(value) for value in feature.shape[-2:]),
        )
        state = torch.cat(
            (
                encoded_feature,
                projected_occupancy.to(dtype=encoded_feature.dtype),
            ),
            dim=1,
        )
        hidden = F.silu(self.input_projection(state))
        hidden = hidden + F.silu(self.spatial_mixing(hidden))
        native_phase = self.phase_projection(hidden)
        field = self.pixel_shuffle(native_phase)
        fields = CoverageStateLevelSetFields(
            encoded_feature=encoded_feature.contiguous(),
            projected_occupancy=projected_occupancy.contiguous(),
            hidden=hidden.contiguous(),
            native_phase_field=native_phase.contiguous(),
            field=field.contiguous(),
            output_size=output_size,
        )
        self._validate_fields(
            fields,
            feature=feature,
            occupancy=occupancy,
        )
        return fields

    def _validate_fields(
        self,
        fields: CoverageStateLevelSetFields,
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
                "projected_occupancy",
                fields.projected_occupancy,
                (batch, 1, height, width),
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
            if name == "projected_occupancy":
                if value.dtype != torch.bool:
                    raise TypeError("projected_occupancy must be bool")
            elif value.dtype != feature.dtype:
                raise TypeError(f"{name} dtype differs from feature")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"{name} must be finite")
    def forward(self, feature: Tensor, occupancy: Tensor) -> Tensor:
        """Return the non-saturating scalar level-set field."""

        return self.forward_fields(feature, occupancy).field

    def predict_completion(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        """Return the fixed negative zero-level-set residual mask."""

        field = self.forward(feature, occupancy)
        return ((field < 0.0) & ~occupancy).contiguous()

    def predict_union(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        """Preserve Base occupancy by exact hard union."""

        return (
            occupancy | self.predict_completion(feature, occupancy)
        ).contiguous()


__all__ = [
    "CSLF_FIELD_POLICY",
    "CSLF_FIELD_AMPLITUDE",
    "CSLF_FEATURE_POLICY",
    "CSLF_INITIAL_FIELD_VALUE",
    "CSLF_NORMALIZATION_EPSILON",
    "CSLF_NUMERICAL_POLICY",
    "CSLF_OUTPUT_POLICY",
    "CSLF_TARGET_POLICY",
    "CURELiteCoverageStateLevelSet",
    "CoverageStateLevelSetConfig",
    "CoverageStateLevelSetFields",
    "truncated_signed_distance_field",
]
