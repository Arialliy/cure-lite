"""Centered mixed-interaction completion field for CURE-Lite v17.

CMIF writes a completion field only through the mixed finite difference
between frozen feature evidence and an exact occupancy phase:

```
phi = a + E(B, U) - E(0, U) - E(B, U_mid_p) + E(0, U_mid_p)
```

``U_mid_p`` differs from the actual phase occupancy only at the current
coarse cell and output phase, where the binary coordinate is replaced by the
fixed analytic midpoint ``0.5``.  The implementation below evaluates the same
equation efficiently through the centre column of one radius-two joint
kernel.  It has one hidden state, one scalar energy, one output field, and no
learned threshold or auxiliary prediction head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .coverage_state_level_set import (
    CSLF_FEATURE_POLICY,
    CSLF_FIELD_AMPLITUDE,
    CSLF_INITIAL_FIELD_VALUE,
    CSLF_NORMALIZATION_EPSILON,
    CSLF_NUMERICAL_POLICY,
    CSLF_OUTPUT_POLICY,
    CSLF_TARGET_POLICY,
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
    normalize_cslf_feature,
)
from .coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
    pixel_unshuffle_bool_occupancy,
)


CSLF_CMIF_FIELD_POLICY = (
    "centered_mixed_finite_difference_exact_phase_radius2_field_v1"
)
CSLF_CMIF_EQUATION_POLICY = (
    "single_shared_silu_joint_energy_feature_and_local_phase_centered_v1"
)
CSLF_CMIF_COARSE_RADIUS = 2
CSLF_CMIF_NEUTRAL_PHASE = 0.5
CSLF_CMIF_KERNEL_SIZE = 2 * CSLF_CMIF_COARSE_RADIUS + 1
CMIF_INTERACTION_POLICY = CSLF_CMIF_FIELD_POLICY
CMIF_ENERGY_POLICY = CSLF_CMIF_EQUATION_POLICY
CMIF_INPUT_REPRESENTATION = "phase_preserving"
CMIF_COARSE_RADIUS = CSLF_CMIF_COARSE_RADIUS
CMIF_NEUTRAL_PHASE = CSLF_CMIF_NEUTRAL_PHASE


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def centered_mixed_energy_difference(
    energy_feature_occupancy: Tensor,
    energy_zero_feature_occupancy: Tensor,
    energy_feature_midpoint: Tensor,
    energy_zero_feature_midpoint: Tensor,
) -> Tensor:
    """Return the ordered four-corner mixed finite difference.

    Keeping this operation explicit makes the algebraic gauge invariances
    directly testable.  All four tensors must be aligned floating tensors;
    broadcasting is deliberately prohibited so that a phase or batch axis
    cannot be silently dropped.
    """

    values = (
        energy_feature_occupancy,
        energy_zero_feature_occupancy,
        energy_feature_midpoint,
        energy_zero_feature_midpoint,
    )
    if (
        any(not isinstance(value, Tensor) for value in values)
        or any(not value.is_floating_point() for value in values)
        or any(value.shape != values[0].shape for value in values[1:])
        or any(value.device != values[0].device for value in values[1:])
        or any(value.dtype != values[0].dtype for value in values[1:])
    ):
        raise TypeError(
            "mixed-difference energies must be aligned floating tensors"
        )
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("mixed-difference energies must be finite")
    return (
        energy_feature_occupancy
        - energy_zero_feature_occupancy
        - energy_feature_midpoint
        + energy_zero_feature_midpoint
    )


@dataclass(frozen=True)
class CoverageStateCenteredMixedInteractionConfig(
    CoverageStateLevelSetConfig
):
    """Frozen structural contract for the v17 CMIF field."""

    coarse_radius: int = CSLF_CMIF_COARSE_RADIUS
    neutral_phase: float = CSLF_CMIF_NEUTRAL_PHASE
    field_policy: str = CSLF_CMIF_FIELD_POLICY
    coverage_policy: str = CSLF_PHASE_PRESERVING_COVERAGE_POLICY
    equation_policy: str = CSLF_CMIF_EQUATION_POLICY
    input_representation: str = CMIF_INPUT_REPRESENTATION
    interaction_policy: str = CMIF_INTERACTION_POLICY
    energy_policy: str = CMIF_ENERGY_POLICY

    def __post_init__(self) -> None:
        # The parent validator fixes the legacy field policy, so this
        # architecture validates the inherited contract explicitly.
        for name in ("feature_channels", "feature_stride", "width"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        frozen_scalars = {
            "normalization_epsilon": CSLF_NORMALIZATION_EPSILON,
            "field_amplitude": CSLF_FIELD_AMPLITUDE,
            "initial_field_value": CSLF_INITIAL_FIELD_VALUE,
            "neutral_phase": CSLF_CMIF_NEUTRAL_PHASE,
        }
        for name, expected in frozen_scalars.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, float)
                or value != expected
            ):
                raise ValueError(f"CMIF fixes {name}")
        if self.initial_field_value != self.field_amplitude:
            raise ValueError(
                "initial_field_value must equal the target-field amplitude"
            )
        if self.coarse_radius != CSLF_CMIF_COARSE_RADIUS:
            raise ValueError("CMIF fixes coarse_radius")
        frozen_policies = {
            "field_policy": CSLF_CMIF_FIELD_POLICY,
            "target_policy": CSLF_TARGET_POLICY,
            "output_policy": CSLF_OUTPUT_POLICY,
            "feature_policy": CSLF_FEATURE_POLICY,
            "numerical_policy": CSLF_NUMERICAL_POLICY,
            "coverage_policy": CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
            "equation_policy": CSLF_CMIF_EQUATION_POLICY,
            "input_representation": CMIF_INPUT_REPRESENTATION,
            "interaction_policy": CMIF_INTERACTION_POLICY,
            "energy_policy": CMIF_ENERGY_POLICY,
        }
        for name, expected in frozen_policies.items():
            if getattr(self, name) != expected:
                raise ValueError(f"CMIF fixes {name}")

    @property
    def kernel_size(self) -> int:
        return 2 * self.coarse_radius + 1

    @property
    def phase_occupancy_channels(self) -> int:
        return self.feature_stride**2

    @property
    def expected_parameter_count(self) -> int:
        joint_state = (
            self.width
            * (self.feature_channels + self.phase_occupancy_channels)
            * self.kernel_size
            * self.kernel_size
        )
        return joint_state + self.width + self.width


@dataclass(frozen=True)
class CoverageStateCenteredMixedInteractionFields:
    """Auditable tensors from one efficient CMIF forward."""

    encoded_feature: Tensor
    phase_occupancy: Tensor
    occupancy_affine: Tensor
    feature_affine: Tensor
    joint_affine: Tensor
    feature_presence_energy: Tensor
    neutralized_feature_presence_energy: Tensor
    neutral_delta: Tensor
    base_feature_contrast: Tensor
    neutral_feature_contrast: Tensor
    mixed_hidden: Tensor
    native_phase_interaction: Tensor
    native_phase_field: Tensor
    field: Tensor
    output_size: tuple[int, int]


class CURELiteCenteredMixedInteractionLevelSet(
    CURELiteCoverageStateLevelSet
):
    """One radius-two, exact-phase, nonseparable completion field."""

    config: CoverageStateCenteredMixedInteractionConfig

    def __init__(
        self,
        config: CoverageStateCenteredMixedInteractionConfig,
    ) -> None:
        if not isinstance(
            config,
            CoverageStateCenteredMixedInteractionConfig,
        ):
            raise TypeError(
                "config must be CoverageStateCenteredMixedInteractionConfig"
            )
        nn.Module.__init__(self)
        self.config = config
        self.joint_state_weight = nn.Parameter(
            torch.empty(
                config.width,
                (
                    config.feature_channels
                    + config.phase_occupancy_channels
                ),
                config.kernel_size,
                config.kernel_size,
                dtype=torch.float32,
            )
        )
        self.joint_hidden_bias = nn.Parameter(
            torch.zeros(config.width, dtype=torch.float32)
        )
        self.scalar_energy_weight = nn.Parameter(
            torch.zeros(config.width, dtype=torch.float32)
        )
        self.pixel_shuffle = nn.PixelShuffle(config.feature_stride)
        self._reset_parameters()
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != config.expected_parameter_count:
            raise AssertionError(
                "CMIF parameter count differs from its contract"
            )

    @property
    def feature_channels(self) -> int:
        return self.config.feature_channels

    @property
    def feature_stride(self) -> int:
        return self.config.feature_stride

    @property
    def feature_weight(self) -> Tensor:
        """Return a view of the feature slice of the one joint kernel."""

        return self.joint_state_weight[
            :, : self.config.feature_channels
        ]

    @property
    def occupancy_weight(self) -> Tensor:
        """Return a view of the phase slice of the one joint kernel."""

        return self.joint_state_weight[
            :, self.config.feature_channels :
        ]

    def _reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.joint_state_weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.joint_hidden_bias)
        nn.init.zeros_(self.scalar_energy_weight)

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
            or feature.dtype != torch.float32
        ):
            raise ValueError(
                "feature must be float32 [B,C,h,w] with configured C"
            )
        if (
            not isinstance(occupancy, Tensor)
            or occupancy.dtype != torch.bool
            or occupancy.ndim != 4
            or occupancy.shape[0] < 1
            or occupancy.shape[1] != 1
            or min(occupancy.shape[-2:]) < 1
        ):
            raise ValueError("occupancy must be bool [B,1,H,W]")
        if feature.shape[0] != occupancy.shape[0]:
            raise ValueError("feature and occupancy batch sizes differ")
        if feature.device != occupancy.device:
            raise ValueError("feature and occupancy must share a device")
        if (
            feature.device != self.joint_state_weight.device
            or self.joint_state_weight.dtype != torch.float32
        ):
            raise ValueError("CMIF model and inputs must be FP32 on one device")
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

    def _affine_states(
        self,
        encoded_feature: Tensor,
        phase_occupancy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        radius = self.config.coarse_radius
        occupancy_affine = F.conv2d(
            phase_occupancy.to(dtype=encoded_feature.dtype),
            self.occupancy_weight,
            bias=self.joint_hidden_bias,
            padding=radius,
        )
        feature_affine = F.conv2d(
            encoded_feature,
            self.feature_weight,
            bias=None,
            padding=radius,
        )
        joint_affine = occupancy_affine + feature_affine
        return occupancy_affine, feature_affine, joint_affine

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CoverageStateCenteredMixedInteractionFields:
        """Evaluate CMIF through the efficient centre-column identity."""

        output_size = self._validate_inputs(feature, occupancy)
        encoded_feature = normalize_cslf_feature(
            feature,
            epsilon=self.config.normalization_epsilon,
        )
        phase_occupancy = pixel_unshuffle_bool_occupancy(
            occupancy,
            stride=self.config.feature_stride,
        )
        (
            occupancy_affine,
            feature_affine,
            joint_affine,
        ) = self._affine_states(encoded_feature, phase_occupancy)

        feature_presence_hidden = (
            F.silu(joint_affine) - F.silu(occupancy_affine)
        )
        feature_presence_energy = (
            feature_presence_hidden
            * self.scalar_energy_weight[None, :, None, None]
        ).sum(dim=1, keepdim=True)

        center = self.config.coarse_radius
        center_phase_weight = self.occupancy_weight[
            :, :, center, center
        ].transpose(0, 1)
        phase_delta = (
            self.config.neutral_phase
            - phase_occupancy.to(dtype=encoded_feature.dtype)
        ).unsqueeze(2) * center_phase_weight[
            None, :, :, None, None
        ]
        midpoint_joint_affine = joint_affine.unsqueeze(1) + phase_delta
        midpoint_occupancy_affine = (
            occupancy_affine.unsqueeze(1) + phase_delta
        )
        neutralized_feature_presence_hidden = (
            F.silu(midpoint_joint_affine)
            - F.silu(midpoint_occupancy_affine)
        )
        neutralized_feature_presence_energy = (
            neutralized_feature_presence_hidden
            * self.scalar_energy_weight[None, None, :, None, None]
        ).sum(dim=2)
        mixed_hidden = (
            feature_presence_hidden.unsqueeze(1)
            - neutralized_feature_presence_hidden
        )
        native_phase_interaction = (
            mixed_hidden
            * self.scalar_energy_weight[None, None, :, None, None]
        ).sum(dim=2)
        native_phase_field = (
            self.config.field_amplitude + native_phase_interaction
        )
        field = self.pixel_shuffle(native_phase_field)
        fields = CoverageStateCenteredMixedInteractionFields(
            encoded_feature=encoded_feature.contiguous(),
            phase_occupancy=phase_occupancy.contiguous(),
            occupancy_affine=occupancy_affine.contiguous(),
            feature_affine=feature_affine.contiguous(),
            joint_affine=joint_affine.contiguous(),
            feature_presence_energy=(
                feature_presence_energy.contiguous()
            ),
            neutralized_feature_presence_energy=(
                neutralized_feature_presence_energy.contiguous()
            ),
            neutral_delta=phase_delta.contiguous(),
            base_feature_contrast=feature_presence_hidden.contiguous(),
            neutral_feature_contrast=(
                neutralized_feature_presence_hidden.contiguous()
            ),
            mixed_hidden=mixed_hidden.contiguous(),
            native_phase_interaction=(
                native_phase_interaction.contiguous()
            ),
            native_phase_field=native_phase_field.contiguous(),
            field=field.contiguous(),
            output_size=output_size,
        )
        self._validate_cmif_fields(
            fields,
            feature=feature,
            occupancy=occupancy,
        )
        return fields

    def forward_reference(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        """Evaluate the explicit local four-corner equation.

        This path is intentionally slow and exists only as a structural
        oracle for small dataset-free probes.  It constructs the midpoint at
        one output cell and one phase at a time.
        """

        self._validate_inputs(feature, occupancy)
        encoded = normalize_cslf_feature(
            feature,
            epsilon=self.config.normalization_epsilon,
        )
        phase = pixel_unshuffle_bool_occupancy(
            occupancy,
            stride=self.config.feature_stride,
        ).to(dtype=encoded.dtype)
        radius = self.config.coarse_radius
        feature_padded = F.pad(
            encoded,
            (radius, radius, radius, radius),
        )
        occupancy_padded = F.pad(
            phase,
            (radius, radius, radius, radius),
        )
        batch, _, height, width = encoded.shape
        native_batches: list[Tensor] = []
        for batch_index in range(batch):
            native_phases: list[Tensor] = []
            for phase_index in range(self.config.phase_channels):
                rows: list[Tensor] = []
                for row in range(height):
                    cells: list[Tensor] = []
                    for column in range(width):
                        feature_patch = feature_padded[
                            batch_index,
                            :,
                            row : row + self.config.kernel_size,
                            column : column + self.config.kernel_size,
                        ]
                        occupancy_patch = occupancy_padded[
                            batch_index,
                            :,
                            row : row + self.config.kernel_size,
                            column : column + self.config.kernel_size,
                        ]
                        midpoint_patch = occupancy_patch.clone()
                        midpoint_patch[
                            phase_index,
                            radius,
                            radius,
                        ] = self.config.neutral_phase
                        cells.append(
                            self.config.field_amplitude
                            + self._explicit_local_interaction(
                                feature_patch,
                                occupancy_patch,
                                midpoint_patch,
                            )
                        )
                    rows.append(torch.stack(cells))
                native_phases.append(torch.stack(rows))
            native_batches.append(torch.stack(native_phases))
        native = torch.stack(native_batches)
        return self.pixel_shuffle(native).contiguous()

    def _explicit_local_interaction(
        self,
        feature_patch: Tensor,
        occupancy_patch: Tensor,
        midpoint_patch: Tensor,
    ) -> Tensor:
        zero_feature = torch.zeros_like(feature_patch)

        def energy(feature_value: Tensor, occupancy_value: Tensor) -> Tensor:
            state = torch.cat((feature_value, occupancy_value), dim=0)
            affine = (
                (
                    self.joint_state_weight
                    * state.unsqueeze(0)
                ).sum(dim=(1, 2, 3))
                + self.joint_hidden_bias
            )
            return (
                F.silu(affine) * self.scalar_energy_weight
            ).sum()

        return centered_mixed_energy_difference(
            energy(feature_patch, occupancy_patch),
            energy(zero_feature, occupancy_patch),
            energy(feature_patch, midpoint_patch),
            energy(zero_feature, midpoint_patch),
        )

    def _validate_cmif_fields(
        self,
        fields: CoverageStateCenteredMixedInteractionFields,
        *,
        feature: Tensor,
        occupancy: Tensor,
    ) -> None:
        batch, _, height, width = feature.shape
        phase_channels = self.config.phase_channels
        expected = (
            ("encoded_feature", fields.encoded_feature, tuple(feature.shape)),
            (
                "phase_occupancy",
                fields.phase_occupancy,
                (batch, phase_channels, height, width),
            ),
            (
                "occupancy_affine",
                fields.occupancy_affine,
                (batch, self.config.width, height, width),
            ),
            (
                "feature_affine",
                fields.feature_affine,
                (batch, self.config.width, height, width),
            ),
            (
                "joint_affine",
                fields.joint_affine,
                (batch, self.config.width, height, width),
            ),
            (
                "feature_presence_energy",
                fields.feature_presence_energy,
                (batch, 1, height, width),
            ),
            (
                "neutralized_feature_presence_energy",
                fields.neutralized_feature_presence_energy,
                (batch, phase_channels, height, width),
            ),
            (
                "neutral_delta",
                fields.neutral_delta,
                (
                    batch,
                    phase_channels,
                    self.config.width,
                    height,
                    width,
                ),
            ),
            (
                "base_feature_contrast",
                fields.base_feature_contrast,
                (batch, self.config.width, height, width),
            ),
            (
                "neutral_feature_contrast",
                fields.neutral_feature_contrast,
                (
                    batch,
                    phase_channels,
                    self.config.width,
                    height,
                    width,
                ),
            ),
            (
                "mixed_hidden",
                fields.mixed_hidden,
                (
                    batch,
                    phase_channels,
                    self.config.width,
                    height,
                    width,
                ),
            ),
            (
                "native_phase_interaction",
                fields.native_phase_interaction,
                (batch, phase_channels, height, width),
            ),
            (
                "native_phase_field",
                fields.native_phase_field,
                (batch, phase_channels, height, width),
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
            elif value.dtype != torch.float32:
                raise TypeError(f"{name} must be float32")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"{name} must be finite")


CURELiteCenteredMixedInteractionField = (
    CURELiteCenteredMixedInteractionLevelSet
)


__all__ = [
    "CMIF_COARSE_RADIUS",
    "CMIF_ENERGY_POLICY",
    "CMIF_INPUT_REPRESENTATION",
    "CMIF_INTERACTION_POLICY",
    "CMIF_NEUTRAL_PHASE",
    "CSLF_CMIF_COARSE_RADIUS",
    "CSLF_CMIF_EQUATION_POLICY",
    "CSLF_CMIF_FIELD_POLICY",
    "CSLF_CMIF_KERNEL_SIZE",
    "CSLF_CMIF_NEUTRAL_PHASE",
    "CURELiteCenteredMixedInteractionField",
    "CURELiteCenteredMixedInteractionLevelSet",
    "CoverageStateCenteredMixedInteractionConfig",
    "CoverageStateCenteredMixedInteractionFields",
    "centered_mixed_energy_difference",
]
