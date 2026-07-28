"""Phase-aligned centered residual compatibility energy for CURE-Lite v22.

PACRE keeps the v21 ``(F_b, O)`` interface, phase transport, binary occupancy
flip, parameter topology, one scalar field, and fixed zero-level-set decoder.
It changes exactly one mathematical object: the reference state of the shared
feature/coverage energy.

For the transported phase feature affine ``A_F^p`` and its phase mean
``bar(A_F)``, PACRE defines

``C_p(O,F) = w^T[
    SiLU(A_U(O) + A_F^p) - SiLU(A_U(O) + bar(A_F))
]``

and retains the binary-flip odd projection

``Delta_p = 0.5 * [C_p(O,F) - C_p(flip_p(O),F)]``.

The phase-common feature is therefore a nonlinear operating point but cannot
alone trigger completion.  Only phase-specific feature residual compatibility
with the counterfactual occupancy change enters the one completion field.
No head, branch, learned threshold, scale, decoder, or post-processing stage
is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CSLF_BFA_COARSE_RADIUS,
)
from cure_lite.coverage_state_level_set import (
    CSLF_FEATURE_POLICY,
    CSLF_FIELD_AMPLITUDE,
    CSLF_INITIAL_FIELD_VALUE,
    CSLF_NORMALIZATION_EPSILON,
    CSLF_NUMERICAL_POLICY,
    CSLF_OUTPUT_POLICY,
    CSLF_TARGET_POLICY,
    normalize_cslf_feature,
)
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CSLF_PAET_FLIP_POLICY,
    CSLF_PAET_TRANSPORT_POLICY,
    PAET_INPUT_REPRESENTATION,
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
    bilinear_phase_aligned_feature_affine,
    row_major_phase_unpack,
)
from cure_lite.coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
    pixel_unshuffle_bool_occupancy,
)


CSLF_PACRE_FIELD_POLICY = (
    "phase_aligned_centered_residual_compatibility_"
    "binary_flip_field_v1"
)
CSLF_PACRE_EQUATION_POLICY = (
    "phase_common_operating_point_specific_residual_"
    "shared_silu_energy_binary_odd_projection_v1"
)
CSLF_PACRE_CENTERING_POLICY = (
    "exact_per_cell_hidden_channel_phase_mean_quotient_v1"
)
PACRE_INTERACTION_POLICY = CSLF_PACRE_FIELD_POLICY
PACRE_ENERGY_POLICY = CSLF_PACRE_EQUATION_POLICY


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def phase_centered_feature_affine(
    phase_feature_affine: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the exact phase mean and phase-specific residual.

    The input has shape ``[B,P,W,h,w]``.  Centering is performed independently
    for every batch item, hidden channel, and coarse spatial cell.
    """

    if (
        not isinstance(phase_feature_affine, Tensor)
        or phase_feature_affine.ndim != 5
        or not phase_feature_affine.is_floating_point()
        or min(phase_feature_affine.shape) < 1
        or not bool(torch.isfinite(phase_feature_affine).all())
    ):
        raise ValueError(
            "phase_feature_affine must be finite floating [B,P,W,h,w]"
        )
    return _phase_centered_feature_affine_unchecked(
        phase_feature_affine
    )


def _phase_centered_feature_affine_unchecked(
    phase_feature_affine: Tensor,
) -> tuple[Tensor, Tensor]:
    """Center one already-validated phase tensor without a device sync."""

    phase_mean = phase_feature_affine.mean(
        dim=1,
        keepdim=True,
    ).contiguous()
    residual = (phase_feature_affine - phase_mean).contiguous()
    return phase_mean, residual


@dataclass(frozen=True)
class CoverageStatePACREConfig(
    CoverageStatePhaseAlignedEvidenceTransportConfig
):
    """Frozen v22 configuration with the exact v21 parameter topology."""

    coarse_radius: int = CSLF_BFA_COARSE_RADIUS
    field_policy: str = CSLF_PACRE_FIELD_POLICY
    coverage_policy: str = CSLF_PHASE_PRESERVING_COVERAGE_POLICY
    equation_policy: str = CSLF_PACRE_EQUATION_POLICY
    flip_policy: str = CSLF_PAET_FLIP_POLICY
    transport_policy: str = CSLF_PAET_TRANSPORT_POLICY
    input_representation: str = PAET_INPUT_REPRESENTATION
    interaction_policy: str = PACRE_INTERACTION_POLICY
    energy_policy: str = PACRE_ENERGY_POLICY
    centering_policy: str = CSLF_PACRE_CENTERING_POLICY

    def __post_init__(self) -> None:
        for name in ("feature_channels", "feature_stride", "width"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        if self.feature_stride < 2:
            raise ValueError(
                "PACRE requires feature_stride >= 2 so that a "
                "nonzero phase-residual subspace exists"
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
                raise ValueError(f"PACRE fixes {name}")
        if self.initial_field_value != self.field_amplitude:
            raise ValueError(
                "initial_field_value must equal the target-field amplitude"
            )
        if self.coarse_radius != CSLF_BFA_COARSE_RADIUS:
            raise ValueError("PACRE fixes coarse_radius")
        frozen_policies = {
            "field_policy": CSLF_PACRE_FIELD_POLICY,
            "target_policy": CSLF_TARGET_POLICY,
            "output_policy": CSLF_OUTPUT_POLICY,
            "feature_policy": CSLF_FEATURE_POLICY,
            "numerical_policy": CSLF_NUMERICAL_POLICY,
            "coverage_policy": CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
            "equation_policy": CSLF_PACRE_EQUATION_POLICY,
            "flip_policy": CSLF_PAET_FLIP_POLICY,
            "transport_policy": CSLF_PAET_TRANSPORT_POLICY,
            "input_representation": PAET_INPUT_REPRESENTATION,
            "interaction_policy": PACRE_INTERACTION_POLICY,
            "energy_policy": PACRE_ENERGY_POLICY,
            "centering_policy": CSLF_PACRE_CENTERING_POLICY,
        }
        for name, expected in frozen_policies.items():
            if getattr(self, name) != expected:
                raise ValueError(f"PACRE fixes {name}")


@dataclass(frozen=True)
class CoverageStatePACREFields:
    """Auditable tensors from one efficient PACRE forward."""

    encoded_feature: Tensor
    phase_occupancy: Tensor
    occupancy_affine: Tensor
    coarse_feature_affine: Tensor
    upsampled_feature_affine: Tensor
    phase_feature_affine: Tensor
    phase_feature_mean: Tensor
    phase_feature_residual: Tensor
    actual_common_joint_affine: Tensor
    actual_specific_joint_affine: Tensor
    actual_compatibility_hidden: Tensor
    actual_compatibility_energy: Tensor
    flip_delta: Tensor
    flipped_occupancy_affine: Tensor
    flipped_common_joint_affine: Tensor
    flipped_specific_joint_affine: Tensor
    flipped_compatibility_hidden: Tensor
    flipped_compatibility_energy: Tensor
    native_phase_interaction: Tensor
    native_phase_field: Tensor
    field: Tensor
    output_size: tuple[int, int]


class CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
    CURELitePhaseAlignedEvidenceTransportLevelSet
):
    """One PACRE field: phase-centered compatibility under a binary flip."""

    config: CoverageStatePACREConfig

    def __init__(self, config: CoverageStatePACREConfig) -> None:
        if type(config) is not CoverageStatePACREConfig:
            raise TypeError("config must be CoverageStatePACREConfig")
        super().__init__(config)

    def _compatibility_energy(
        self,
        occupancy_affine: Tensor,
        phase_feature_affine: Tensor,
        phase_feature_mean: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        common_joint = (
            occupancy_affine + phase_feature_mean
        ).expand_as(phase_feature_affine)
        specific_joint = occupancy_affine + phase_feature_affine
        hidden = F.silu(specific_joint) - F.silu(common_joint)
        energy = (
            hidden
            * self.scalar_energy_weight[None, None, :, None, None]
        ).sum(dim=2)
        return (
            common_joint.contiguous(),
            specific_joint.contiguous(),
            hidden.contiguous(),
            energy.contiguous(),
        )

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CoverageStatePACREFields:
        """Evaluate PACRE and every center-phase binary flip."""

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
        phase_feature_mean, phase_feature_residual = (
            _phase_centered_feature_affine_unchecked(
                phase_feature_affine
            )
        )
        (
            actual_common_joint_affine,
            actual_specific_joint_affine,
            actual_compatibility_hidden,
            actual_compatibility_energy,
        ) = self._compatibility_energy(
            occupancy_affine.unsqueeze(1),
            phase_feature_affine,
            phase_feature_mean,
        )

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
        (
            flipped_common_joint_affine,
            flipped_specific_joint_affine,
            flipped_compatibility_hidden,
            flipped_compatibility_energy,
        ) = self._compatibility_energy(
            flipped_occupancy_affine,
            phase_feature_affine,
            phase_feature_mean,
        )
        native_phase_interaction = 0.5 * (
            actual_compatibility_energy
            - flipped_compatibility_energy
        )
        native_phase_field = (
            self.config.field_amplitude + native_phase_interaction
        )
        field = self.pixel_shuffle(native_phase_field)
        fields = CoverageStatePACREFields(
            encoded_feature=encoded_feature.contiguous(),
            phase_occupancy=phase_occupancy.contiguous(),
            occupancy_affine=occupancy_affine.contiguous(),
            coarse_feature_affine=coarse_feature_affine.contiguous(),
            upsampled_feature_affine=(
                upsampled_feature_affine.contiguous()
            ),
            phase_feature_affine=phase_feature_affine.contiguous(),
            phase_feature_mean=phase_feature_mean.contiguous(),
            phase_feature_residual=phase_feature_residual.contiguous(),
            actual_common_joint_affine=(
                actual_common_joint_affine.contiguous()
            ),
            actual_specific_joint_affine=(
                actual_specific_joint_affine.contiguous()
            ),
            actual_compatibility_hidden=(
                actual_compatibility_hidden.contiguous()
            ),
            actual_compatibility_energy=(
                actual_compatibility_energy.contiguous()
            ),
            flip_delta=flip_delta.contiguous(),
            flipped_occupancy_affine=(
                flipped_occupancy_affine.contiguous()
            ),
            flipped_common_joint_affine=(
                flipped_common_joint_affine.contiguous()
            ),
            flipped_specific_joint_affine=(
                flipped_specific_joint_affine.contiguous()
            ),
            flipped_compatibility_hidden=(
                flipped_compatibility_hidden.contiguous()
            ),
            flipped_compatibility_energy=(
                flipped_compatibility_energy.contiguous()
            ),
            native_phase_interaction=(
                native_phase_interaction.contiguous()
            ),
            native_phase_field=native_phase_field.contiguous(),
            field=field.contiguous(),
            output_size=output_size,
        )
        self._validate_pacre_fields(
            fields,
            feature=feature,
            occupancy=occupancy,
        )
        return fields

    def _explicit_local_compatibility_energy(
        self,
        phase_feature_affine: Tensor,
        phase_feature_mean: Tensor,
        occupancy_patch: Tensor,
    ) -> Tensor:
        occupancy_affine = (
            (
                self.occupancy_weight
                * occupancy_patch.to(
                    dtype=phase_feature_affine.dtype
                ).unsqueeze(0)
            ).sum(dim=(1, 2, 3))
            + self.joint_hidden_bias
        )
        hidden = (
            F.silu(occupancy_affine + phase_feature_affine)
            - F.silu(occupancy_affine + phase_feature_mean)
        )
        return (hidden * self.scalar_energy_weight).sum()

    def forward_reference(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        """Evaluate the literal local PACRE equation as an oracle."""

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
        phase_mean, _ = _phase_centered_feature_affine_unchecked(
            phase_feature
        )
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
                        flipped_patch = occupancy_patch.clone()
                        flipped_patch[
                            phase_index,
                            radius,
                            radius,
                        ] = ~flipped_patch[
                            phase_index,
                            radius,
                            radius,
                        ]
                        actual = (
                            self._explicit_local_compatibility_energy(
                                phase_feature[
                                    batch_index,
                                    phase_index,
                                    :,
                                    row,
                                    column,
                                ],
                                phase_mean[
                                    batch_index,
                                    0,
                                    :,
                                    row,
                                    column,
                                ],
                                occupancy_patch,
                            )
                        )
                        flipped = (
                            self._explicit_local_compatibility_energy(
                                phase_feature[
                                    batch_index,
                                    phase_index,
                                    :,
                                    row,
                                    column,
                                ],
                                phase_mean[
                                    batch_index,
                                    0,
                                    :,
                                    row,
                                    column,
                                ],
                                flipped_patch,
                            )
                        )
                        cells.append(
                            self.config.field_amplitude
                            + 0.5 * (actual - flipped)
                        )
                    rows.append(torch.stack(cells))
                native_phases.append(torch.stack(rows))
            native_batches.append(torch.stack(native_phases))
        return self.pixel_shuffle(torch.stack(native_batches)).contiguous()

    def _validate_pacre_fields(
        self,
        fields: CoverageStatePACREFields,
        *,
        feature: Tensor,
        occupancy: Tensor,
    ) -> None:
        batch, _, height, width = feature.shape
        phases = self.config.phase_channels
        hidden = self.config.width
        output_height, output_width = occupancy.shape[-2:]
        phase_hidden = (batch, phases, hidden, height, width)
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
                phase_hidden,
            ),
            (
                "phase_feature_mean",
                fields.phase_feature_mean,
                (batch, 1, hidden, height, width),
            ),
            (
                "phase_feature_residual",
                fields.phase_feature_residual,
                phase_hidden,
            ),
            (
                "actual_common_joint_affine",
                fields.actual_common_joint_affine,
                phase_hidden,
            ),
            (
                "actual_specific_joint_affine",
                fields.actual_specific_joint_affine,
                phase_hidden,
            ),
            (
                "actual_compatibility_hidden",
                fields.actual_compatibility_hidden,
                phase_hidden,
            ),
            (
                "actual_compatibility_energy",
                fields.actual_compatibility_energy,
                (batch, phases, height, width),
            ),
            ("flip_delta", fields.flip_delta, phase_hidden),
            (
                "flipped_occupancy_affine",
                fields.flipped_occupancy_affine,
                phase_hidden,
            ),
            (
                "flipped_common_joint_affine",
                fields.flipped_common_joint_affine,
                phase_hidden,
            ),
            (
                "flipped_specific_joint_affine",
                fields.flipped_specific_joint_affine,
                phase_hidden,
            ),
            (
                "flipped_compatibility_hidden",
                fields.flipped_compatibility_hidden,
                phase_hidden,
            ),
            (
                "flipped_compatibility_energy",
                fields.flipped_compatibility_energy,
                (batch, phases, height, width),
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
            raise FloatingPointError("PACRE fields must be finite")
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
        residual_sum = fields.phase_feature_residual.sum(dim=1)
        residual_tolerance = (
            4.0
            * float(phases)
            * torch.finfo(torch.float32).eps
            * (
                1.0
                + fields.phase_feature_affine.detach().abs().amax()
            )
        )
        if bool(torch.any(residual_sum.abs() > residual_tolerance)):
            raise AssertionError("phase residual is not centered")


CURELitePhaseAlignedCenteredResidualCompatibilityEnergyField = (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
)


__all__ = [
    "CSLF_PACRE_CENTERING_POLICY",
    "CSLF_PACRE_EQUATION_POLICY",
    "CSLF_PACRE_FIELD_POLICY",
    "CURELitePhaseAlignedCenteredResidualCompatibilityEnergyField",
    "CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet",
    "CoverageStatePACREConfig",
    "CoverageStatePACREFields",
    "phase_centered_feature_affine",
]
