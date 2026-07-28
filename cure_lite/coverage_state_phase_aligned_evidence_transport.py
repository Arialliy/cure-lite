"""Phase-aligned evidence transport for the CURE-Lite BFA field.

PAET changes only where the already projected feature evidence is evaluated.
The coarse feature affine state is bilinearly transported to the output grid
and packed back into the row-major PixelShuffle phase basis.  Every phase then
uses its own aligned feature affine state in the same shared BFA energy:

``A_F = conv(B, W_F)``
``A_F^p = phase_pack(interpolate(A_F, scale_factor=s))``
``G_p(B,U) = w^T[SiLU(A_U + A_F^p) - SiLU(A_U)]``
``Delta_p = 0.5 [G_p(B,U) - G_p(B, flip_p(U))]``
``phi = PixelShuffle(a + Delta)``.

There is one completion field and no additional parameter, prediction head,
branch, curvature term, or learned scale.  Parameter names, shapes,
initialization, and count are exactly those of BFA-CMIF.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .coverage_state_binary_flip_antisymmetric import (
    CSLF_BFA_COARSE_RADIUS,
    CSLF_BFA_KERNEL_SIZE,
    binary_flip_odd_projection,
    flip_binary_center_phase,
)
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


CSLF_PAET_FIELD_POLICY = (
    "phase_aligned_evidence_transport_binary_flip_field_v1"
)
CSLF_PAET_EQUATION_POLICY = (
    "bilinear_phase_aligned_shared_silu_energy_binary_odd_projection_v1"
)
CSLF_PAET_TRANSPORT_POLICY = (
    "align_corners_false_bilinear_then_row_major_phase_pack_v1"
)
CSLF_PAET_FLIP_POLICY = "exact_binary_current_center_phase_involution_v1"
PAET_INPUT_REPRESENTATION = "phase_preserving"
PAET_COARSE_RADIUS = CSLF_BFA_COARSE_RADIUS
PAET_INTERACTION_POLICY = CSLF_PAET_FIELD_POLICY
PAET_ENERGY_POLICY = CSLF_PAET_EQUATION_POLICY
PAET_TRANSPORT_POLICY = CSLF_PAET_TRANSPORT_POLICY
PAET_FLIP_POLICY = CSLF_PAET_FLIP_POLICY


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def align_corners_false_axis_offsets(stride: int) -> tuple[float, ...]:
    """Return fine-sample offsets along one coarse-cell axis."""

    stride = _positive_int(stride, name="stride")
    return tuple(
        (2 * index + 1 - stride) / (2.0 * stride)
        for index in range(stride)
    )


def align_corners_false_phase_offsets(
    stride: int,
) -> tuple[tuple[float, float], ...]:
    """Return ``(row,column)`` offsets in row-major phase order."""

    one_axis = align_corners_false_axis_offsets(stride)
    return tuple(
        (row_offset, column_offset)
        for row_offset in one_axis
        for column_offset in one_axis
    )


def row_major_phase_pack(
    fine_grid: Tensor,
    *,
    stride: int,
) -> Tensor:
    """Pack ``[B,W,h*s,w*s]`` as row-major ``[B,s**2,W,h,w]``.

    PyTorch PixelUnshuffle groups phases after each input channel.  The
    reshape/permute below changes only axis order so that phase is explicit
    and first, while preserving its canonical row-major convention.
    """

    if (
        not isinstance(fine_grid, Tensor)
        or not fine_grid.is_floating_point()
        or fine_grid.ndim != 4
        or min(fine_grid.shape) < 1
    ):
        raise TypeError("fine_grid must be floating [B,W,H,W]")
    stride = _positive_int(stride, name="stride")
    batch, width, height, columns = fine_grid.shape
    if height % stride != 0 or columns % stride != 0:
        raise ValueError("fine_grid spatial dimensions must divide by stride")
    packed = F.pixel_unshuffle(fine_grid, stride)
    return (
        packed.reshape(
            batch,
            width,
            stride * stride,
            height // stride,
            columns // stride,
        )
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def row_major_phase_unpack(
    phase_grid: Tensor,
    *,
    stride: int,
) -> Tensor:
    """Invert :func:`row_major_phase_pack` exactly."""

    if (
        not isinstance(phase_grid, Tensor)
        or not phase_grid.is_floating_point()
        or phase_grid.ndim != 5
        or min(phase_grid.shape) < 1
    ):
        raise TypeError("phase_grid must be floating [B,P,W,h,w]")
    stride = _positive_int(stride, name="stride")
    batch, phases, width, height, columns = phase_grid.shape
    if phases != stride * stride:
        raise ValueError("phase axis must equal stride squared")
    native = (
        phase_grid.permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(
            batch,
            width * phases,
            height,
            columns,
        )
    )
    return F.pixel_shuffle(native, stride).contiguous()


def bilinear_phase_aligned_feature_affine(
    coarse_feature_affine: Tensor,
    *,
    stride: int,
) -> tuple[Tensor, Tensor]:
    """Transport a coarse affine state and expose its aligned phase pack."""

    if (
        not isinstance(coarse_feature_affine, Tensor)
        or not coarse_feature_affine.is_floating_point()
        or coarse_feature_affine.ndim != 4
        or min(coarse_feature_affine.shape) < 1
    ):
        raise TypeError(
            "coarse_feature_affine must be floating [B,W,h,w]"
        )
    stride = _positive_int(stride, name="stride")
    upsampled = F.interpolate(
        coarse_feature_affine,
        scale_factor=stride,
        mode="bilinear",
        align_corners=False,
    ).contiguous()
    phase = row_major_phase_pack(upsampled, stride=stride)
    return upsampled, phase


@dataclass(frozen=True)
class CoverageStatePhaseAlignedEvidenceTransportConfig(
    CoverageStateLevelSetConfig
):
    """Frozen PAET-BFA structure with exactly the BFA parameter contract."""

    coarse_radius: int = CSLF_BFA_COARSE_RADIUS
    field_policy: str = CSLF_PAET_FIELD_POLICY
    coverage_policy: str = CSLF_PHASE_PRESERVING_COVERAGE_POLICY
    equation_policy: str = CSLF_PAET_EQUATION_POLICY
    flip_policy: str = CSLF_PAET_FLIP_POLICY
    transport_policy: str = CSLF_PAET_TRANSPORT_POLICY
    input_representation: str = PAET_INPUT_REPRESENTATION
    interaction_policy: str = PAET_INTERACTION_POLICY
    energy_policy: str = PAET_ENERGY_POLICY

    def __post_init__(self) -> None:
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
                raise ValueError(f"PAET-BFA fixes {name}")
        if self.initial_field_value != self.field_amplitude:
            raise ValueError(
                "initial_field_value must equal the target-field amplitude"
            )
        if self.coarse_radius != CSLF_BFA_COARSE_RADIUS:
            raise ValueError("PAET-BFA fixes coarse_radius")
        frozen_policies = {
            "field_policy": CSLF_PAET_FIELD_POLICY,
            "target_policy": CSLF_TARGET_POLICY,
            "output_policy": CSLF_OUTPUT_POLICY,
            "feature_policy": CSLF_FEATURE_POLICY,
            "numerical_policy": CSLF_NUMERICAL_POLICY,
            "coverage_policy": CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
            "equation_policy": CSLF_PAET_EQUATION_POLICY,
            "flip_policy": CSLF_PAET_FLIP_POLICY,
            "transport_policy": CSLF_PAET_TRANSPORT_POLICY,
            "input_representation": PAET_INPUT_REPRESENTATION,
            "interaction_policy": PAET_INTERACTION_POLICY,
            "energy_policy": PAET_ENERGY_POLICY,
        }
        for name, expected in frozen_policies.items():
            if getattr(self, name) != expected:
                raise ValueError(f"PAET-BFA fixes {name}")

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
class CoverageStatePhaseAlignedEvidenceTransportFields:
    """Auditable tensors from one efficient PAET-BFA forward."""

    encoded_feature: Tensor
    phase_occupancy: Tensor
    occupancy_affine: Tensor
    coarse_feature_affine: Tensor
    upsampled_feature_affine: Tensor
    phase_feature_affine: Tensor
    phase_joint_affine: Tensor
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


class CURELitePhaseAlignedEvidenceTransportLevelSet(
    CURELiteCoverageStateLevelSet
):
    """One PAET-BFA completion field with deterministic phase transport."""

    config: CoverageStatePhaseAlignedEvidenceTransportConfig

    def __init__(
        self,
        config: CoverageStatePhaseAlignedEvidenceTransportConfig,
    ) -> None:
        if not isinstance(
            config,
            CoverageStatePhaseAlignedEvidenceTransportConfig,
        ):
            raise TypeError(
                "config must be CoverageStatePhaseAlignedEvidenceTransportConfig"
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
                "PAET-BFA parameter count differs from its contract"
            )

    @property
    def feature_channels(self) -> int:
        return self.config.feature_channels

    @property
    def feature_stride(self) -> int:
        return self.config.feature_stride

    @property
    def feature_weight(self) -> Tensor:
        return self.joint_state_weight[
            :, : self.config.feature_channels
        ]

    @property
    def occupancy_weight(self) -> Tensor:
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
                "PAET-BFA model and inputs must be FP32 on one device"
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        radius = self.config.coarse_radius
        occupancy_affine = F.conv2d(
            phase_occupancy.to(dtype=encoded_feature.dtype),
            self.occupancy_weight,
            bias=self.joint_hidden_bias,
            padding=radius,
        )
        coarse_feature_affine = F.conv2d(
            encoded_feature,
            self.feature_weight,
            bias=None,
            padding=radius,
        )
        upsampled, phase = bilinear_phase_aligned_feature_affine(
            coarse_feature_affine,
            stride=self.config.feature_stride,
        )
        return occupancy_affine, coarse_feature_affine, upsampled, phase

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CoverageStatePhaseAlignedEvidenceTransportFields:
        """Evaluate PAET and every binary flip without another convolution."""

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
            coarse_feature_affine,
            upsampled_feature_affine,
            phase_feature_affine,
        ) = self._affine_states(encoded_feature, phase_occupancy)
        phase_joint_affine = (
            occupancy_affine.unsqueeze(1) + phase_feature_affine
        )
        actual_feature_presence_hidden = (
            F.silu(phase_joint_affine)
            - F.silu(occupancy_affine.unsqueeze(1))
        )
        actual_feature_presence_energy = (
            actual_feature_presence_hidden
            * self.scalar_energy_weight[None, None, :, None, None]
        ).sum(dim=2)

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
        flipped_joint_affine = (
            flipped_occupancy_affine + phase_feature_affine
        )
        flipped_feature_presence_hidden = (
            F.silu(flipped_joint_affine)
            - F.silu(flipped_occupancy_affine)
        )
        flipped_feature_presence_energy = (
            flipped_feature_presence_hidden
            * self.scalar_energy_weight[None, None, :, None, None]
        ).sum(dim=2)
        odd_feature_presence_hidden = 0.5 * (
            actual_feature_presence_hidden
            - flipped_feature_presence_hidden
        )
        native_phase_interaction = 0.5 * (
            actual_feature_presence_energy
            - flipped_feature_presence_energy
        )
        native_phase_field = (
            self.config.field_amplitude + native_phase_interaction
        )
        field = self.pixel_shuffle(native_phase_field)
        fields = CoverageStatePhaseAlignedEvidenceTransportFields(
            encoded_feature=encoded_feature.contiguous(),
            phase_occupancy=phase_occupancy.contiguous(),
            occupancy_affine=occupancy_affine.contiguous(),
            coarse_feature_affine=coarse_feature_affine.contiguous(),
            upsampled_feature_affine=(
                upsampled_feature_affine.contiguous()
            ),
            phase_feature_affine=phase_feature_affine.contiguous(),
            phase_joint_affine=phase_joint_affine.contiguous(),
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
        self._validate_paet_fields(
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
        """Evaluate the literal phase-aligned local energy equation."""

        self._validate_inputs(feature, occupancy)
        encoded = normalize_cslf_feature(
            feature,
            epsilon=self.config.normalization_epsilon,
        )
        phase = pixel_unshuffle_bool_occupancy(
            occupancy,
            stride=self.config.feature_stride,
        )
        _, _, _, phase_feature = self._affine_states(encoded, phase)
        radius = self.config.coarse_radius
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
                                phase_feature[
                                    batch_index,
                                    phase_index,
                                    :,
                                    row,
                                    column,
                                ],
                                occupancy_patch,
                                flipped_patch,
                            )
                        )
                    rows.append(torch.stack(cells))
                native_phases.append(torch.stack(rows))
            native_batches.append(torch.stack(native_phases))
        return self.pixel_shuffle(torch.stack(native_batches)).contiguous()

    def _explicit_local_feature_presence_energy(
        self,
        feature_affine: Tensor,
        occupancy_patch: Tensor,
    ) -> Tensor:
        occupancy_affine = (
            (
                self.occupancy_weight
                * occupancy_patch.to(
                    dtype=feature_affine.dtype
                ).unsqueeze(0)
            ).sum(dim=(1, 2, 3))
            + self.joint_hidden_bias
        )
        hidden = (
            F.silu(occupancy_affine + feature_affine)
            - F.silu(occupancy_affine)
        )
        return (hidden * self.scalar_energy_weight).sum()

    def _explicit_local_interaction(
        self,
        feature_affine: Tensor,
        occupancy_patch: Tensor,
        flipped_patch: Tensor,
    ) -> Tensor:
        actual = self._explicit_local_feature_presence_energy(
            feature_affine,
            occupancy_patch,
        )
        flipped = self._explicit_local_feature_presence_energy(
            feature_affine,
            flipped_patch,
        )
        return binary_flip_odd_projection(actual, flipped)

    def _validate_paet_fields(
        self,
        fields: CoverageStatePhaseAlignedEvidenceTransportFields,
        *,
        feature: Tensor,
        occupancy: Tensor,
    ) -> None:
        batch, _, height, width = feature.shape
        phases = self.config.phase_channels
        hidden = self.config.width
        output_height, output_width = occupancy.shape[-2:]
        expected = (
            ("encoded_feature", fields.encoded_feature, tuple(feature.shape)),
            (
                "phase_occupancy",
                fields.phase_occupancy,
                (batch, phases, height, width),
            ),
            (
                "occupancy_affine",
                fields.occupancy_affine,
                (batch, hidden, height, width),
            ),
            (
                "coarse_feature_affine",
                fields.coarse_feature_affine,
                (batch, hidden, height, width),
            ),
            (
                "upsampled_feature_affine",
                fields.upsampled_feature_affine,
                (batch, hidden, output_height, output_width),
            ),
            (
                "phase_feature_affine",
                fields.phase_feature_affine,
                (batch, phases, hidden, height, width),
            ),
            (
                "phase_joint_affine",
                fields.phase_joint_affine,
                (batch, phases, hidden, height, width),
            ),
            (
                "actual_feature_presence_hidden",
                fields.actual_feature_presence_hidden,
                (batch, phases, hidden, height, width),
            ),
            (
                "actual_feature_presence_energy",
                fields.actual_feature_presence_energy,
                (batch, phases, height, width),
            ),
            (
                "flip_delta",
                fields.flip_delta,
                (batch, phases, hidden, height, width),
            ),
            (
                "flipped_occupancy_affine",
                fields.flipped_occupancy_affine,
                (batch, phases, hidden, height, width),
            ),
            (
                "flipped_joint_affine",
                fields.flipped_joint_affine,
                (batch, phases, hidden, height, width),
            ),
            (
                "flipped_feature_presence_hidden",
                fields.flipped_feature_presence_hidden,
                (batch, phases, hidden, height, width),
            ),
            (
                "flipped_feature_presence_energy",
                fields.flipped_feature_presence_energy,
                (batch, phases, height, width),
            ),
            (
                "odd_feature_presence_hidden",
                fields.odd_feature_presence_hidden,
                (batch, phases, hidden, height, width),
            ),
            (
                "native_phase_interaction",
                fields.native_phase_interaction,
                (batch, phases, height, width),
            ),
            (
                "native_phase_field",
                fields.native_phase_field,
                (batch, phases, height, width),
            ),
            ("field", fields.field, tuple(occupancy.shape)),
        )
        finite: list[Tensor] = []
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
            finite.append(torch.isfinite(value).all())
        if not bool(torch.stack(finite).all()):
            raise FloatingPointError("PAET-BFA fields must be finite")
        if fields.output_size != tuple(occupancy.shape[-2:]):
            raise AssertionError("output_size differs from occupancy")
        reconstructed = row_major_phase_unpack(
            fields.phase_feature_affine,
            stride=self.config.feature_stride,
        )
        if not torch.equal(
            reconstructed,
            fields.upsampled_feature_affine,
        ):
            raise AssertionError("phase transport roundtrip changed evidence")


CURELitePhaseAlignedEvidenceTransportField = (
    CURELitePhaseAlignedEvidenceTransportLevelSet
)


__all__ = [
    "CSLF_PAET_EQUATION_POLICY",
    "CSLF_PAET_FIELD_POLICY",
    "CSLF_PAET_FLIP_POLICY",
    "CSLF_PAET_TRANSPORT_POLICY",
    "CURELitePhaseAlignedEvidenceTransportField",
    "CURELitePhaseAlignedEvidenceTransportLevelSet",
    "CoverageStatePhaseAlignedEvidenceTransportConfig",
    "CoverageStatePhaseAlignedEvidenceTransportFields",
    "PAET_COARSE_RADIUS",
    "PAET_ENERGY_POLICY",
    "PAET_FLIP_POLICY",
    "PAET_INPUT_REPRESENTATION",
    "PAET_INTERACTION_POLICY",
    "PAET_TRANSPORT_POLICY",
    "align_corners_false_axis_offsets",
    "align_corners_false_phase_offsets",
    "bilinear_phase_aligned_feature_affine",
    "row_major_phase_pack",
    "row_major_phase_unpack",
]
