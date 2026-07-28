"""Binary-flip antisymmetric completion field for CURE-Lite.

The field is the odd projection of one shared scalar energy under the
involution that flips only the current output phase at the current coarse
cell.  For the feature-presence energy

``H(B, U) = E(B, U) - E(0, U)``,

the interaction and field are

``Delta(B, U) = 0.5 * (H(B, U) - H(B, flip(U)))``

and

``phi(B, U) = a + Delta(B, U)``.

The flip leaves every other occupancy coordinate unchanged.  Consequently
the interaction is exactly antisymmetric with respect to the selected binary
coordinate, while feature-only and occupancy-only paths cancel.  The module
uses the same three parameter tensors, shapes, initialization, radius-two
joint kernel, and scalar energy as CMIF; it adds no learned component.
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


CSLF_BFA_FIELD_POLICY = (
    "binary_flip_antisymmetric_feature_presence_radius2_field_v1"
)
CSLF_BFA_EQUATION_POLICY = (
    "single_shared_silu_joint_energy_center_phase_z2_odd_projection_v1"
)
CSLF_BFA_FLIP_POLICY = (
    "exact_binary_current_center_phase_involution_v1"
)
CSLF_BFA_COARSE_RADIUS = 2
CSLF_BFA_KERNEL_SIZE = 2 * CSLF_BFA_COARSE_RADIUS + 1
BFA_INTERACTION_POLICY = CSLF_BFA_FIELD_POLICY
BFA_ENERGY_POLICY = CSLF_BFA_EQUATION_POLICY
BFA_FLIP_POLICY = CSLF_BFA_FLIP_POLICY
BFA_INPUT_REPRESENTATION = "phase_preserving"
BFA_COARSE_RADIUS = CSLF_BFA_COARSE_RADIUS


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def binary_flip_odd_projection(
    actual_feature_presence_energy: Tensor,
    flipped_feature_presence_energy: Tensor,
) -> Tensor:
    """Return the odd projection under an aligned binary flip.

    Broadcasting is prohibited so that a missing phase, batch, or spatial
    axis cannot silently change the intended local group action.
    """

    values = (
        actual_feature_presence_energy,
        flipped_feature_presence_energy,
    )
    if (
        any(not isinstance(value, Tensor) for value in values)
        or any(not value.is_floating_point() for value in values)
        or values[0].shape != values[1].shape
        or values[0].device != values[1].device
        or values[0].dtype != values[1].dtype
    ):
        raise TypeError(
            "binary-flip energies must be aligned floating tensors"
        )
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("binary-flip energies must be finite")
    return 0.5 * (values[0] - values[1])


def flip_binary_center_phase(
    phase_occupancy_patch: Tensor,
    *,
    phase_index: int,
    center: int,
) -> Tensor:
    """Flip exactly one Boolean phase coordinate in a local patch."""

    if (
        not isinstance(phase_occupancy_patch, Tensor)
        or phase_occupancy_patch.dtype != torch.bool
        or phase_occupancy_patch.ndim != 3
        or min(phase_occupancy_patch.shape) < 1
    ):
        raise TypeError(
            "phase_occupancy_patch must be bool [P,K_h,K_w]"
        )
    if (
        isinstance(phase_index, bool)
        or not isinstance(phase_index, int)
        or not 0 <= phase_index < phase_occupancy_patch.shape[0]
    ):
        raise ValueError("phase_index is outside the phase axis")
    if (
        isinstance(center, bool)
        or not isinstance(center, int)
        or not 0 <= center < phase_occupancy_patch.shape[-2]
        or not 0 <= center < phase_occupancy_patch.shape[-1]
    ):
        raise ValueError("center is outside the spatial patch")
    flipped = phase_occupancy_patch.clone()
    flipped[phase_index, center, center] = torch.logical_not(
        flipped[phase_index, center, center]
    )
    return flipped.contiguous()


@dataclass(frozen=True)
class CoverageStateBinaryFlipAntisymmetricConfig(
    CoverageStateLevelSetConfig
):
    """Frozen structural contract for the BFA-CMIF field."""

    coarse_radius: int = CSLF_BFA_COARSE_RADIUS
    field_policy: str = CSLF_BFA_FIELD_POLICY
    coverage_policy: str = CSLF_PHASE_PRESERVING_COVERAGE_POLICY
    equation_policy: str = CSLF_BFA_EQUATION_POLICY
    flip_policy: str = CSLF_BFA_FLIP_POLICY
    input_representation: str = BFA_INPUT_REPRESENTATION
    interaction_policy: str = BFA_INTERACTION_POLICY
    energy_policy: str = BFA_ENERGY_POLICY

    def __post_init__(self) -> None:
        # The parent validator freezes the legacy field policy, so the
        # independent BFA architecture validates the inherited fields here.
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
        }
        for name, expected in frozen_scalars.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, float)
                or value != expected
            ):
                raise ValueError(f"BFA-CMIF fixes {name}")
        if self.initial_field_value != self.field_amplitude:
            raise ValueError(
                "initial_field_value must equal the target-field amplitude"
            )
        if self.coarse_radius != CSLF_BFA_COARSE_RADIUS:
            raise ValueError("BFA-CMIF fixes coarse_radius")
        frozen_policies = {
            "field_policy": CSLF_BFA_FIELD_POLICY,
            "target_policy": CSLF_TARGET_POLICY,
            "output_policy": CSLF_OUTPUT_POLICY,
            "feature_policy": CSLF_FEATURE_POLICY,
            "numerical_policy": CSLF_NUMERICAL_POLICY,
            "coverage_policy": CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
            "equation_policy": CSLF_BFA_EQUATION_POLICY,
            "flip_policy": CSLF_BFA_FLIP_POLICY,
            "input_representation": BFA_INPUT_REPRESENTATION,
            "interaction_policy": BFA_INTERACTION_POLICY,
            "energy_policy": BFA_ENERGY_POLICY,
        }
        for name, expected in frozen_policies.items():
            if getattr(self, name) != expected:
                raise ValueError(f"BFA-CMIF fixes {name}")

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
class CoverageStateBinaryFlipAntisymmetricFields:
    """Auditable tensors from one efficient BFA-CMIF forward."""

    encoded_feature: Tensor
    phase_occupancy: Tensor
    occupancy_affine: Tensor
    feature_affine: Tensor
    joint_affine: Tensor
    actual_feature_presence_hidden: Tensor
    actual_feature_presence_energy: Tensor
    flip_delta: Tensor
    flipped_occupancy_affine: Tensor
    flipped_joint_affine: Tensor
    flipped_feature_presence_hidden: Tensor
    flipped_feature_presence_energy: Tensor
    odd_feature_presence_hidden: Tensor
    native_phase_interaction: Tensor
    native_phase_field: Tensor
    field: Tensor
    output_size: tuple[int, int]


class CURELiteBinaryFlipAntisymmetricLevelSet(
    CURELiteCoverageStateLevelSet
):
    """One radius-two scalar-energy field with a local binary odd projection."""

    config: CoverageStateBinaryFlipAntisymmetricConfig

    def __init__(
        self,
        config: CoverageStateBinaryFlipAntisymmetricConfig,
    ) -> None:
        if not isinstance(
            config,
            CoverageStateBinaryFlipAntisymmetricConfig,
        ):
            raise TypeError(
                "config must be CoverageStateBinaryFlipAntisymmetricConfig"
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
                "BFA-CMIF parameter count differs from its contract"
            )

    @property
    def feature_channels(self) -> int:
        return self.config.feature_channels

    @property
    def feature_stride(self) -> int:
        return self.config.feature_stride

    @property
    def feature_weight(self) -> Tensor:
        """Return a view of the feature slice of the joint kernel."""

        return self.joint_state_weight[
            :, : self.config.feature_channels
        ]

    @property
    def occupancy_weight(self) -> Tensor:
        """Return a view of the phase-occupancy slice of the joint kernel."""

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
            raise ValueError(
                "BFA-CMIF model and inputs must be FP32 on one device"
            )
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
        return (
            occupancy_affine,
            feature_affine,
            occupancy_affine + feature_affine,
        )

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CoverageStateBinaryFlipAntisymmetricFields:
        """Evaluate the local binary odd projection without a second convolution."""

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

        actual_feature_presence_hidden = (
            F.silu(joint_affine) - F.silu(occupancy_affine)
        )
        actual_feature_presence_energy = (
            actual_feature_presence_hidden
            * self.scalar_energy_weight[None, :, None, None]
        ).sum(dim=1, keepdim=True)

        center = self.config.coarse_radius
        center_phase_weight = self.occupancy_weight[
            :, :, center, center
        ].transpose(0, 1)
        flip_delta = (
            1.0
            - 2.0 * phase_occupancy.to(dtype=encoded_feature.dtype)
        ).unsqueeze(2) * center_phase_weight[
            None, :, :, None, None
        ]
        flipped_occupancy_affine = (
            occupancy_affine.unsqueeze(1) + flip_delta
        )
        flipped_joint_affine = joint_affine.unsqueeze(1) + flip_delta
        flipped_feature_presence_hidden = (
            F.silu(flipped_joint_affine)
            - F.silu(flipped_occupancy_affine)
        )
        flipped_feature_presence_energy = (
            flipped_feature_presence_hidden
            * self.scalar_energy_weight[None, None, :, None, None]
        ).sum(dim=2)
        odd_feature_presence_hidden = 0.5 * (
            actual_feature_presence_hidden.unsqueeze(1)
            - flipped_feature_presence_hidden
        )
        actual_phase_energy = actual_feature_presence_energy.expand_as(
            flipped_feature_presence_energy
        )
        # The aligned tensor contract is established by the shapes above.
        # Keep the public checked helper for external probes, but avoid its
        # finite-reduction synchronization in every training forward.
        native_phase_interaction = 0.5 * (
            actual_phase_energy - flipped_feature_presence_energy
        )
        native_phase_field = (
            self.config.field_amplitude + native_phase_interaction
        )
        field = self.pixel_shuffle(native_phase_field)
        fields = CoverageStateBinaryFlipAntisymmetricFields(
            encoded_feature=encoded_feature.contiguous(),
            phase_occupancy=phase_occupancy.contiguous(),
            occupancy_affine=occupancy_affine.contiguous(),
            feature_affine=feature_affine.contiguous(),
            joint_affine=joint_affine.contiguous(),
            actual_feature_presence_hidden=(
                actual_feature_presence_hidden.contiguous()
            ),
            actual_feature_presence_energy=(
                actual_feature_presence_energy.contiguous()
            ),
            flip_delta=flip_delta.contiguous(),
            flipped_occupancy_affine=(
                flipped_occupancy_affine.contiguous()
            ),
            flipped_joint_affine=flipped_joint_affine.contiguous(),
            flipped_feature_presence_hidden=(
                flipped_feature_presence_hidden.contiguous()
            ),
            flipped_feature_presence_energy=(
                flipped_feature_presence_energy.contiguous()
            ),
            odd_feature_presence_hidden=(
                odd_feature_presence_hidden.contiguous()
            ),
            native_phase_interaction=(
                native_phase_interaction.contiguous()
            ),
            native_phase_field=native_phase_field.contiguous(),
            field=field.contiguous(),
            output_size=output_size,
        )
        self._validate_bfa_fields(
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
        """Evaluate the literal local energy equation as a structural oracle."""

        self._validate_inputs(feature, occupancy)
        encoded = normalize_cslf_feature(
            feature,
            epsilon=self.config.normalization_epsilon,
        )
        phase = pixel_unshuffle_bool_occupancy(
            occupancy,
            stride=self.config.feature_stride,
        )
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
                        flipped_patch = flip_binary_center_phase(
                            occupancy_patch,
                            phase_index=phase_index,
                            center=radius,
                        )
                        cells.append(
                            self.config.field_amplitude
                            + self._explicit_local_interaction(
                                feature_patch,
                                occupancy_patch,
                                flipped_patch,
                            )
                        )
                    rows.append(torch.stack(cells))
                native_phases.append(torch.stack(rows))
            native_batches.append(torch.stack(native_phases))
        native = torch.stack(native_batches)
        return self.pixel_shuffle(native).contiguous()

    def _explicit_local_feature_presence_energy(
        self,
        feature_patch: Tensor,
        occupancy_patch: Tensor,
    ) -> Tensor:
        zero_feature = torch.zeros_like(feature_patch)

        def energy(feature_value: Tensor) -> Tensor:
            state = torch.cat(
                (
                    feature_value,
                    occupancy_patch.to(dtype=feature_value.dtype),
                ),
                dim=0,
            )
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

        return energy(feature_patch) - energy(zero_feature)

    def _explicit_local_interaction(
        self,
        feature_patch: Tensor,
        occupancy_patch: Tensor,
        flipped_patch: Tensor,
    ) -> Tensor:
        actual = self._explicit_local_feature_presence_energy(
            feature_patch,
            occupancy_patch,
        )
        flipped = self._explicit_local_feature_presence_energy(
            feature_patch,
            flipped_patch,
        )
        return binary_flip_odd_projection(actual, flipped)

    def _validate_bfa_fields(
        self,
        fields: CoverageStateBinaryFlipAntisymmetricFields,
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
                "actual_feature_presence_hidden",
                fields.actual_feature_presence_hidden,
                (batch, self.config.width, height, width),
            ),
            (
                "actual_feature_presence_energy",
                fields.actual_feature_presence_energy,
                (batch, 1, height, width),
            ),
            (
                "flip_delta",
                fields.flip_delta,
                (
                    batch,
                    phase_channels,
                    self.config.width,
                    height,
                    width,
                ),
            ),
            (
                "flipped_occupancy_affine",
                fields.flipped_occupancy_affine,
                (
                    batch,
                    phase_channels,
                    self.config.width,
                    height,
                    width,
                ),
            ),
            (
                "flipped_joint_affine",
                fields.flipped_joint_affine,
                (
                    batch,
                    phase_channels,
                    self.config.width,
                    height,
                    width,
                ),
            ),
            (
                "flipped_feature_presence_hidden",
                fields.flipped_feature_presence_hidden,
                (
                    batch,
                    phase_channels,
                    self.config.width,
                    height,
                    width,
                ),
            ),
            (
                "flipped_feature_presence_energy",
                fields.flipped_feature_presence_energy,
                (batch, phase_channels, height, width),
            ),
            (
                "odd_feature_presence_hidden",
                fields.odd_feature_presence_hidden,
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
        finite_checks: list[Tensor] = []
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
            finite_checks.append(torch.isfinite(value).all())
        # One host synchronization is sufficient for the complete field
        # ledger; synchronizing once per intermediate made every GPU
        # training forward unnecessarily serial.
        if not bool(torch.stack(finite_checks).all()):
            raise FloatingPointError("BFA-CMIF fields must be finite")

        if fields.output_size != tuple(occupancy.shape[-2:]):
            raise AssertionError("output_size differs from occupancy")


CURELiteBinaryFlipAntisymmetricField = (
    CURELiteBinaryFlipAntisymmetricLevelSet
)


__all__ = [
    "BFA_COARSE_RADIUS",
    "BFA_ENERGY_POLICY",
    "BFA_FLIP_POLICY",
    "BFA_INPUT_REPRESENTATION",
    "BFA_INTERACTION_POLICY",
    "CSLF_BFA_COARSE_RADIUS",
    "CSLF_BFA_EQUATION_POLICY",
    "CSLF_BFA_FIELD_POLICY",
    "CSLF_BFA_FLIP_POLICY",
    "CSLF_BFA_KERNEL_SIZE",
    "CURELiteBinaryFlipAntisymmetricField",
    "CURELiteBinaryFlipAntisymmetricLevelSet",
    "CoverageStateBinaryFlipAntisymmetricConfig",
    "CoverageStateBinaryFlipAntisymmetricFields",
    "binary_flip_odd_projection",
    "flip_binary_center_phase",
]
