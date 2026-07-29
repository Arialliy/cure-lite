"""Gated common-residual PACRE field for CURE-Lite v24.

GCR-PACRE keeps the v22/v23 input interface, parameter tensors,
initialization, phase transport, binary center-phase flip, and fixed
zero-level-set decoder.  It changes only the scalar interaction:

``D_p = 0.5 * (R_p(O) - R_p(flip_p(O)))``
``E_p = 0.5 * (C_p(O) + C_p(flip_p(O)))``
``G_p = 2 * sigmoid(E_p)``
``I_p = G_p * D_p``
``phi_p = 0.9 + I_p``

Here ``R`` is the v22 phase-specific centered-residual energy and ``C`` is
the phase-common feature-presence energy.  Consequently ``D`` and ``I`` are
odd under the designated binary flip, while ``E`` and ``G`` are even.

The mathematical sigmoid range is open, but finite FP32 evaluation can
round to either endpoint.  The executable machine contract is therefore
finite ``0 <= G <= 2``.  Endpoint saturation is retained in the auditable
fields rather than rejected as an invalid state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

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
    row_major_phase_unpack,
)
from cure_lite.coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
    pixel_unshuffle_bool_occupancy,
)


GCR_PACRE_CANDIDATE: Final = "GCR-PACRE-v24"
GCR_PACRE_METHOD_ID: Final = "cure_lite_gcr_pacre_v24"
CSLF_GCR_PACRE_FIELD_POLICY: Final = (
    "gcr_pacre_single_zero_level_set_field_v1"
)
CSLF_GCR_PACRE_EQUATION_POLICY: Final = (
    "flip_even_common_gate_times_flip_odd_residual_v1"
)
GCR_PACRE_INTERACTION_POLICY: Final = (
    "bounded_even_gate_times_binary_flip_odd_residual_v1"
)
GCR_PACRE_ENERGY_POLICY: Final = (
    "shared_readout_residual_and_common_compatibility_v1"
)
GCR_PACRE_NUMERICAL_POLICY: Final = (
    "finite_closed_gate_interval_with_saturation_audit_v1"
)
GCR_PACRE_CENTERING_POLICY: Final = (
    "exact_per_cell_hidden_channel_phase_mean_quotient_v1"
)
GCR_PACRE_GATE_STATISTICS_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-gate-statistics-v1"
)
GCR_PACRE_FP64_ORACLE_ABS_TOL: Final = 2.0e-6
GCR_PACRE_FP64_ORACLE_MAX_ULP: Final = 32
GCR_PACRE_FIELDS_FQCN: Final = (
    "cure_lite_v24.gcr_pacre.CoverageStateGCRPACREFields"
)


def _phase_centered_feature_affine(
    phase_feature_affine: Tensor,
) -> tuple[Tensor, Tensor]:
    """Center a validated ``[B,P,W,h,w]`` tensor in the phase axis."""

    phase_mean = phase_feature_affine.mean(
        dim=1,
        keepdim=True,
    ).contiguous()
    phase_residual = (
        phase_feature_affine - phase_mean
    ).contiguous()
    return phase_mean, phase_residual


@dataclass(frozen=True)
class CoverageStateGCRPACREConfig(
    CoverageStatePhaseAlignedEvidenceTransportConfig
):
    """Canonical v24 identity on the unchanged PAET parameter topology."""

    method_id: str = GCR_PACRE_METHOD_ID
    field_policy: str = CSLF_GCR_PACRE_FIELD_POLICY
    equation_policy: str = CSLF_GCR_PACRE_EQUATION_POLICY
    interaction_policy: str = GCR_PACRE_INTERACTION_POLICY
    energy_policy: str = GCR_PACRE_ENERGY_POLICY
    numerical_policy: str = GCR_PACRE_NUMERICAL_POLICY
    centering_policy: str = GCR_PACRE_CENTERING_POLICY

    def __post_init__(self) -> None:
        for name in ("feature_channels", "feature_stride", "width"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        if self.feature_stride < 2:
            raise ValueError("GCR-PACRE requires feature_stride >= 2")
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
                raise ValueError(f"GCR-PACRE fixes {name}")
        if self.initial_field_value != self.field_amplitude:
            raise ValueError(
                "initial_field_value must equal field_amplitude"
            )
        if (
            isinstance(self.coarse_radius, bool)
            or not isinstance(self.coarse_radius, int)
            or self.coarse_radius != CSLF_BFA_COARSE_RADIUS
        ):
            raise ValueError("GCR-PACRE fixes coarse_radius")
        frozen_policies = {
            "method_id": GCR_PACRE_METHOD_ID,
            "field_policy": CSLF_GCR_PACRE_FIELD_POLICY,
            "target_policy": CSLF_TARGET_POLICY,
            "output_policy": CSLF_OUTPUT_POLICY,
            "feature_policy": CSLF_FEATURE_POLICY,
            "numerical_policy": GCR_PACRE_NUMERICAL_POLICY,
            "coverage_policy": CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
            "equation_policy": CSLF_GCR_PACRE_EQUATION_POLICY,
            "flip_policy": CSLF_PAET_FLIP_POLICY,
            "transport_policy": CSLF_PAET_TRANSPORT_POLICY,
            "input_representation": PAET_INPUT_REPRESENTATION,
            "interaction_policy": GCR_PACRE_INTERACTION_POLICY,
            "energy_policy": GCR_PACRE_ENERGY_POLICY,
            "centering_policy": GCR_PACRE_CENTERING_POLICY,
        }
        for name, expected in frozen_policies.items():
            if getattr(self, name) != expected:
                raise ValueError(f"GCR-PACRE fixes {name}")


@dataclass(frozen=True)
class GCRPACREGateSaturationAudit:
    """Exact endpoint-saturation summary for one native gate tensor."""

    schema: str
    element_count: int
    zero_count: int
    two_count: int
    interior_count: int
    saturated_count: int
    zero_fraction: float
    two_fraction: float
    interior_fraction: float
    saturated_fraction: float
    minimum: float
    maximum: float
    mean: float

    def __post_init__(self) -> None:
        if self.schema != GCR_PACRE_GATE_STATISTICS_SCHEMA:
            raise ValueError("gate statistics schema changed")
        integer_values = (
            self.element_count,
            self.zero_count,
            self.two_count,
            self.interior_count,
            self.saturated_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_values
        ):
            raise TypeError("gate saturation counts must be integers")
        if (
            self.element_count < 1
            or self.zero_count < 0
            or self.two_count < 0
            or self.interior_count < 0
            or self.saturated_count
            != self.zero_count + self.two_count
            or self.saturated_count + self.interior_count
            != self.element_count
        ):
            raise ValueError("invalid gate saturation counts")
        expected_fractions = {
            "zero_fraction": self.zero_count / self.element_count,
            "two_fraction": self.two_count / self.element_count,
            "interior_fraction": (
                self.interior_count / self.element_count
            ),
            "saturated_fraction": (
                self.saturated_count / self.element_count
            ),
        }
        for name, expected in expected_fractions.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, float)
                or not math.isfinite(value)
                or value != expected
            ):
                raise ValueError(f"invalid gate statistic {name}")
        for name in ("minimum", "maximum", "mean"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"invalid gate statistic {name}")
        if (
            self.minimum < 0.0
            or self.maximum > 2.0
            or self.minimum > self.mean
            or self.mean > self.maximum
        ):
            raise ValueError("gate statistics violate [0,2]")


@dataclass(frozen=True)
class CoverageStateGCRPACREFields:
    """Complete auditable state from one efficient GCR-PACRE forward."""

    encoded_feature: Tensor
    phase_occupancy: Tensor
    occupancy_affine: Tensor
    coarse_feature_affine: Tensor
    upsampled_feature_affine: Tensor
    phase_feature_affine: Tensor
    phase_feature_mean: Tensor
    phase_feature_residual: Tensor
    actual_occupancy_only_joint_affine: Tensor
    actual_common_joint_affine: Tensor
    actual_specific_joint_affine: Tensor
    actual_common_silu: Tensor
    actual_residual_hidden: Tensor
    actual_residual_energy: Tensor
    actual_common_hidden: Tensor
    actual_common_energy: Tensor
    center_phase_weight: Tensor
    flip_delta: Tensor
    flipped_center_phase_value: Tensor
    flipped_occupancy_affine: Tensor
    flipped_occupancy_only_joint_affine: Tensor
    flipped_common_joint_affine: Tensor
    flipped_specific_joint_affine: Tensor
    flipped_common_silu: Tensor
    flipped_residual_hidden: Tensor
    flipped_residual_energy: Tensor
    flipped_common_hidden: Tensor
    flipped_common_energy: Tensor
    residual_odd_interaction: Tensor
    common_even_energy: Tensor
    common_gate: Tensor
    common_gate_zero_saturation: Tensor
    common_gate_two_saturation: Tensor
    gated_interaction: Tensor
    native_phase_field: Tensor
    field: Tensor
    output_size: tuple[int, int]


@dataclass(frozen=True)
class GCRPACREFP64OracleFields:
    """Minimal independent FP64 oracle state for algebra and parity checks."""

    residual_odd_interaction: Tensor
    common_even_energy: Tensor
    common_gate: Tensor
    gated_interaction: Tensor
    native_phase_field: Tensor
    field: Tensor
    output_size: tuple[int, int]


@dataclass(frozen=True)
class GCRPACREFP64Comparison:
    """Frozen FP32-fast versus independent-FP64-oracle error report."""

    maximum_absolute_error: float
    maximum_ulp_distance: int
    absolute_tolerance: float
    maximum_allowed_ulp: int
    passed: bool


def gcr_pacre_fp32_ulp_distance(
    actual: Tensor,
    reference_fp64: Tensor,
) -> Tensor:
    """Return elementwise ULP distance to the FP64 value rounded to FP32."""

    if (
        not isinstance(actual, Tensor)
        or actual.dtype != torch.float32
        or not isinstance(reference_fp64, Tensor)
        or reference_fp64.dtype != torch.float64
        or actual.shape != reference_fp64.shape
        or actual.device != reference_fp64.device
    ):
        raise TypeError(
            "ULP comparison requires same-shape/device FP32 and FP64 tensors"
        )
    if not bool(
        torch.stack(
            (
                torch.isfinite(actual).all(),
                torch.isfinite(reference_fp64).all(),
            )
        ).all()
    ):
        raise FloatingPointError("ULP comparison requires finite tensors")

    rounded = reference_fp64.to(dtype=torch.float32)
    actual_canonical = torch.where(
        actual == 0.0,
        torch.zeros_like(actual),
        actual,
    ).contiguous()
    rounded_canonical = torch.where(
        rounded == 0.0,
        torch.zeros_like(rounded),
        rounded,
    ).contiguous()

    def ordered_bits(value: Tensor) -> Tensor:
        unsigned = (
            value.view(torch.int32).to(dtype=torch.int64)
            & 0xFFFFFFFF
        )
        negative = (unsigned & 0x80000000) != 0
        return torch.where(
            negative,
            0xFFFFFFFF - unsigned,
            unsigned + 0x80000000,
        )

    return (
        ordered_bits(actual_canonical)
        - ordered_bits(rounded_canonical)
    ).abs().contiguous()


def compare_gcr_pacre_fp32_to_fp64_oracle(
    actual: Tensor,
    reference_fp64: Tensor,
) -> GCRPACREFP64Comparison:
    """Compare one FP32 field against the frozen v24 oracle envelope."""

    ulp = gcr_pacre_fp32_ulp_distance(actual, reference_fp64)
    maximum_absolute_error = float(
        (
            actual.to(dtype=torch.float64) - reference_fp64
        ).abs().amax().item()
    )
    maximum_ulp_distance = int(ulp.amax().item())
    return GCRPACREFP64Comparison(
        maximum_absolute_error=maximum_absolute_error,
        maximum_ulp_distance=maximum_ulp_distance,
        absolute_tolerance=GCR_PACRE_FP64_ORACLE_ABS_TOL,
        maximum_allowed_ulp=GCR_PACRE_FP64_ORACLE_MAX_ULP,
        passed=(
            maximum_absolute_error
            <= GCR_PACRE_FP64_ORACLE_ABS_TOL
            and maximum_ulp_distance
            <= GCR_PACRE_FP64_ORACLE_MAX_ULP
        ),
    )


def _gate_saturation_masks(gate: Tensor) -> tuple[Tensor, Tensor]:
    """Create auditable endpoint masks without a device synchronization."""

    return (gate == 0.0).contiguous(), (gate == 2.0).contiguous()


def summarize_gcr_pacre_gate_saturation(
    fields: CoverageStateGCRPACREFields,
) -> GCRPACREGateSaturationAudit:
    """Materialize endpoint counts explicitly outside the training forward."""

    if type(fields) is not CoverageStateGCRPACREFields:
        raise TypeError("fields must be CoverageStateGCRPACREFields")
    expected_zero, expected_two = _gate_saturation_masks(
        fields.common_gate
    )
    if (
        not torch.equal(
            fields.common_gate_zero_saturation,
            expected_zero,
        )
        or not torch.equal(
            fields.common_gate_two_saturation,
            expected_two,
        )
    ):
        raise AssertionError(
            "gate saturation masks do not match common_gate"
        )
    zero_count = int(
        torch.count_nonzero(
            fields.common_gate_zero_saturation
        ).item()
    )
    two_count = int(
        torch.count_nonzero(
            fields.common_gate_two_saturation
        ).item()
    )
    element_count = fields.common_gate.numel()
    saturated_count = zero_count + two_count
    interior_count = element_count - saturated_count
    gate = fields.common_gate.detach()
    return GCRPACREGateSaturationAudit(
        schema=GCR_PACRE_GATE_STATISTICS_SCHEMA,
        element_count=element_count,
        zero_count=zero_count,
        two_count=two_count,
        interior_count=interior_count,
        saturated_count=saturated_count,
        zero_fraction=zero_count / element_count,
        two_fraction=two_count / element_count,
        interior_fraction=interior_count / element_count,
        saturated_fraction=saturated_count / element_count,
        minimum=float(gate.amin().item()),
        maximum=float(gate.amax().item()),
        mean=float(gate.mean().item()),
    )


class CURELiteGatedCommonResidualPACRELevelSet(
    CURELitePhaseAlignedEvidenceTransportLevelSet
):
    """One GCR-PACRE field with no parameter or decoder addition."""

    config: CoverageStateGCRPACREConfig

    def __init__(self, config: CoverageStateGCRPACREConfig) -> None:
        if type(config) is not CoverageStateGCRPACREConfig:
            raise TypeError("config must be CoverageStateGCRPACREConfig")
        super().__init__(config)

    def _compatibility_components(
        self,
        occupancy_affine: Tensor,
        phase_feature_affine: Tensor,
        phase_feature_mean: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        occupancy_only_joint = occupancy_affine.expand_as(
            phase_feature_affine
        )
        common_joint = (
            occupancy_affine + phase_feature_mean
        ).expand_as(phase_feature_affine)
        specific_joint = occupancy_affine + phase_feature_affine
        common_silu = F.silu(common_joint)
        residual_hidden = F.silu(specific_joint) - common_silu
        common_hidden = common_silu - F.silu(occupancy_only_joint)
        readout = self.scalar_energy_weight[
            None, None, :, None, None
        ]
        residual_energy = (residual_hidden * readout).sum(dim=2)
        common_energy = (common_hidden * readout).sum(dim=2)
        return (
            occupancy_only_joint.contiguous(),
            common_joint.contiguous(),
            specific_joint.contiguous(),
            common_silu.contiguous(),
            residual_hidden.contiguous(),
            residual_energy.contiguous(),
            common_hidden.contiguous(),
            common_energy.contiguous(),
        )

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CoverageStateGCRPACREFields:
        """Evaluate the GCR equation and expose every replay state."""

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
            _phase_centered_feature_affine(
                phase_feature_affine
            )
        )
        (
            actual_occupancy_only_joint_affine,
            actual_common_joint_affine,
            actual_specific_joint_affine,
            actual_common_silu,
            actual_residual_hidden,
            actual_residual_energy,
            actual_common_hidden,
            actual_common_energy,
        ) = self._compatibility_components(
            occupancy_affine.unsqueeze(1),
            phase_feature_affine,
            phase_feature_mean,
        )

        center = self.config.coarse_radius
        center_phase_weight = self.occupancy_weight[
            :, :, center, center
        ].transpose(0, 1).contiguous()
        flipped_center_phase_value = (
            ~phase_occupancy
        ).contiguous()
        flip_delta = (
            flipped_center_phase_value.to(
                dtype=encoded_feature.dtype
            )
            - phase_occupancy.to(dtype=encoded_feature.dtype)
        ).unsqueeze(2) * center_phase_weight[
            None, :, :, None, None
        ]
        flipped_occupancy_affine = (
            occupancy_affine.unsqueeze(1) + flip_delta
        )
        (
            flipped_occupancy_only_joint_affine,
            flipped_common_joint_affine,
            flipped_specific_joint_affine,
            flipped_common_silu,
            flipped_residual_hidden,
            flipped_residual_energy,
            flipped_common_hidden,
            flipped_common_energy,
        ) = self._compatibility_components(
            flipped_occupancy_affine,
            phase_feature_affine,
            phase_feature_mean,
        )

        residual_odd_interaction = 0.5 * (
            actual_residual_energy - flipped_residual_energy
        )
        common_even_energy = 0.5 * (
            actual_common_energy + flipped_common_energy
        )
        common_gate = 2.0 * torch.sigmoid(common_even_energy)
        (
            common_gate_zero_saturation,
            common_gate_two_saturation,
        ) = _gate_saturation_masks(common_gate)
        gated_interaction = (
            common_gate * residual_odd_interaction
        )
        native_phase_field = (
            self.config.field_amplitude + gated_interaction
        )
        field = self.pixel_shuffle(native_phase_field)

        fields = CoverageStateGCRPACREFields(
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
            actual_occupancy_only_joint_affine=(
                actual_occupancy_only_joint_affine.contiguous()
            ),
            actual_common_joint_affine=(
                actual_common_joint_affine.contiguous()
            ),
            actual_specific_joint_affine=(
                actual_specific_joint_affine.contiguous()
            ),
            actual_common_silu=actual_common_silu.contiguous(),
            actual_residual_hidden=(
                actual_residual_hidden.contiguous()
            ),
            actual_residual_energy=(
                actual_residual_energy.contiguous()
            ),
            actual_common_hidden=(
                actual_common_hidden.contiguous()
            ),
            actual_common_energy=(
                actual_common_energy.contiguous()
            ),
            center_phase_weight=center_phase_weight.contiguous(),
            flip_delta=flip_delta.contiguous(),
            flipped_center_phase_value=(
                flipped_center_phase_value.contiguous()
            ),
            flipped_occupancy_affine=(
                flipped_occupancy_affine.contiguous()
            ),
            flipped_occupancy_only_joint_affine=(
                flipped_occupancy_only_joint_affine.contiguous()
            ),
            flipped_common_joint_affine=(
                flipped_common_joint_affine.contiguous()
            ),
            flipped_specific_joint_affine=(
                flipped_specific_joint_affine.contiguous()
            ),
            flipped_common_silu=flipped_common_silu.contiguous(),
            flipped_residual_hidden=(
                flipped_residual_hidden.contiguous()
            ),
            flipped_residual_energy=(
                flipped_residual_energy.contiguous()
            ),
            flipped_common_hidden=(
                flipped_common_hidden.contiguous()
            ),
            flipped_common_energy=(
                flipped_common_energy.contiguous()
            ),
            residual_odd_interaction=(
                residual_odd_interaction.contiguous()
            ),
            common_even_energy=(
                common_even_energy.contiguous()
            ),
            common_gate=common_gate.contiguous(),
            common_gate_zero_saturation=(
                common_gate_zero_saturation.contiguous()
            ),
            common_gate_two_saturation=(
                common_gate_two_saturation.contiguous()
            ),
            gated_interaction=(
                gated_interaction.contiguous()
            ),
            native_phase_field=native_phase_field.contiguous(),
            field=field.contiguous(),
            output_size=output_size,
        )
        self._validate_gcr_fields(
            fields,
            feature=feature,
            occupancy=occupancy,
        )
        return fields

    def forward_reference_fields_fp64(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> GCRPACREFP64OracleFields:
        """Independently recompute the literal GCR equation in FP64."""

        output_size = self._validate_inputs(feature, occupancy)
        with torch.no_grad():
            stride = self.config.feature_stride
            phases = self.config.phase_channels
            radius = self.config.coarse_radius
            batch, _, height, width = feature.shape

            feature_fp64 = feature.detach().to(dtype=torch.float64)
            sample_rms = feature_fp64.square().mean(
                dim=(1, 2, 3),
                keepdim=True,
            ).sqrt()
            encoded_fp64 = (
                feature_fp64
                / sample_rms.clamp_min(
                    self.config.normalization_epsilon
                )
            ).contiguous()

            phase_occupancy = F.pixel_unshuffle(
                occupancy.to(dtype=torch.float64),
                stride,
            ).to(dtype=torch.bool).contiguous()

            joint_weight = self.joint_state_weight.detach().to(
                dtype=torch.float64
            )
            feature_weight = joint_weight[
                :, : self.config.feature_channels
            ]
            occupancy_weight = joint_weight[
                :, self.config.feature_channels :
            ]
            hidden_bias = self.joint_hidden_bias.detach().to(
                dtype=torch.float64
            )
            readout = self.scalar_energy_weight.detach().to(
                dtype=torch.float64
            )

            coarse_feature = F.conv2d(
                encoded_fp64,
                feature_weight,
                bias=None,
                padding=radius,
            )
            fine_feature = F.interpolate(
                coarse_feature,
                scale_factor=stride,
                mode="bilinear",
                align_corners=False,
            )
            packed_feature = F.pixel_unshuffle(
                fine_feature,
                stride,
            )
            phase_feature = (
                packed_feature.reshape(
                    batch,
                    self.config.width,
                    phases,
                    height,
                    width,
                )
                .permute(0, 2, 1, 3, 4)
                .contiguous()
            )
            phase_mean = phase_feature.mean(
                dim=1,
                keepdim=True,
            ).contiguous()
            occupancy_padded = F.pad(
                phase_occupancy,
                (radius, radius, radius, radius),
            )

            residual_batches: list[Tensor] = []
            common_batches: list[Tensor] = []
            for batch_index in range(batch):
                residual_phases: list[Tensor] = []
                common_phases: list[Tensor] = []
                for phase_index in range(phases):
                    residual_rows: list[Tensor] = []
                    common_rows: list[Tensor] = []
                    for row in range(height):
                        residual_cells: list[Tensor] = []
                        common_cells: list[Tensor] = []
                        for column in range(width):
                            patch = occupancy_padded[
                                batch_index,
                                :,
                                row : row + self.config.kernel_size,
                                column : column + self.config.kernel_size,
                            ]
                            flipped_patch = patch.clone()
                            flipped_patch[
                                phase_index,
                                radius,
                                radius,
                            ] = ~flipped_patch[
                                phase_index,
                                radius,
                                radius,
                            ]
                            feature_cell = phase_feature[
                                batch_index,
                                phase_index,
                                :,
                                row,
                                column,
                            ]
                            mean_cell = phase_mean[
                                batch_index,
                                0,
                                :,
                                row,
                                column,
                            ]
                            local_energies: list[tuple[Tensor, Tensor]] = []
                            for local_patch in (patch, flipped_patch):
                                occupancy_affine = (
                                    (
                                        occupancy_weight
                                        * local_patch.to(
                                            dtype=torch.float64
                                        ).unsqueeze(0)
                                    ).sum(dim=(1, 2, 3))
                                    + hidden_bias
                                )
                                common_silu = F.silu(
                                    occupancy_affine + mean_cell
                                )
                                residual_hidden = (
                                    F.silu(
                                        occupancy_affine + feature_cell
                                    )
                                    - common_silu
                                )
                                common_hidden = (
                                    common_silu
                                    - F.silu(occupancy_affine)
                                )
                                local_energies.append(
                                    (
                                        (
                                            residual_hidden * readout
                                        ).sum(),
                                        (
                                            common_hidden * readout
                                        ).sum(),
                                    )
                                )
                            actual_energy, flipped_energy = local_energies
                            residual_cells.append(
                                0.5
                                * (
                                    actual_energy[0]
                                    - flipped_energy[0]
                                )
                            )
                            common_cells.append(
                                0.5
                                * (
                                    actual_energy[1]
                                    + flipped_energy[1]
                                )
                            )
                        residual_rows.append(
                            torch.stack(residual_cells)
                        )
                        common_rows.append(torch.stack(common_cells))
                    residual_phases.append(torch.stack(residual_rows))
                    common_phases.append(torch.stack(common_rows))
                residual_batches.append(torch.stack(residual_phases))
                common_batches.append(torch.stack(common_phases))

            residual_odd = torch.stack(residual_batches).contiguous()
            common_even = torch.stack(common_batches).contiguous()
            common_gate = (
                2.0 * torch.sigmoid(common_even)
            ).contiguous()
            gated_interaction = (
                common_gate * residual_odd
            ).contiguous()
            native_phase_field = (
                float(self.config.field_amplitude)
                + gated_interaction
            ).contiguous()
            field = F.pixel_shuffle(
                native_phase_field,
                stride,
            ).contiguous()
            return GCRPACREFP64OracleFields(
                residual_odd_interaction=residual_odd,
                common_even_energy=common_even,
                common_gate=common_gate,
                gated_interaction=gated_interaction,
                native_phase_field=native_phase_field,
                field=field,
                output_size=output_size,
            )

    def forward_reference(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        """Return the independent FP64 literal-equation oracle field."""

        return self.forward_reference_fields_fp64(
            feature,
            occupancy,
        ).field

    def forward_forced_unit_gate(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        """Read-only same-weight ``G=1`` mechanism-ablation field."""

        fields = self.forward_fields(feature, occupancy)
        return self.pixel_shuffle(
            self.config.field_amplitude
            + fields.residual_odd_interaction
        ).contiguous()

    def _validate_gcr_fields(
        self,
        fields: CoverageStateGCRPACREFields,
        *,
        feature: Tensor,
        occupancy: Tensor,
    ) -> None:
        _validate_gcr_pacre_fields_contract(
            self,
            fields,
            feature=feature,
            occupancy=occupancy,
            inputs_already_validated=True,
        )


def _require_exact_tensor(
    *,
    name: str,
    actual: Tensor,
    expected: Tensor,
) -> None:
    if not torch.equal(actual, expected):
        raise AssertionError(f"{name} failed exact forward replay")


def _validate_gcr_pacre_fields_contract(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    fields: CoverageStateGCRPACREFields,
    *,
    feature: Tensor,
    occupancy: Tensor,
    inputs_already_validated: bool,
) -> None:
    """Validate the training-time fields contract with one device sync."""

    if type(model) is not CURELiteGatedCommonResidualPACRELevelSet:
        raise TypeError("model must be the exact GCR-PACRE v24 type")
    if type(fields) is not CoverageStateGCRPACREFields:
        raise TypeError("fields must be CoverageStateGCRPACREFields")
    identity = {
        "method_id": GCR_PACRE_METHOD_ID,
        "field_policy": CSLF_GCR_PACRE_FIELD_POLICY,
        "equation_policy": CSLF_GCR_PACRE_EQUATION_POLICY,
        "interaction_policy": GCR_PACRE_INTERACTION_POLICY,
        "energy_policy": GCR_PACRE_ENERGY_POLICY,
        "numerical_policy": GCR_PACRE_NUMERICAL_POLICY,
    }
    if any(
        getattr(model.config, name) != expected
        for name, expected in identity.items()
    ):
        raise ValueError("GCR-PACRE policy identity changed")
    output_size = (
        tuple(int(value) for value in occupancy.shape[-2:])
        if inputs_already_validated
        else model._validate_inputs(feature, occupancy)
    )
    batch, _, height, width = feature.shape
    phases = model.config.phase_channels
    hidden = model.config.width
    output_height, output_width = occupancy.shape[-2:]
    phase_hidden = (batch, phases, hidden, height, width)
    phase_scalar = (batch, phases, height, width)
    expected_shapes = (
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
        ("phase_feature_affine", fields.phase_feature_affine, phase_hidden),
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
            "actual_occupancy_only_joint_affine",
            fields.actual_occupancy_only_joint_affine,
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
            "actual_common_silu",
            fields.actual_common_silu,
            phase_hidden,
        ),
        (
            "actual_residual_hidden",
            fields.actual_residual_hidden,
            phase_hidden,
        ),
        (
            "actual_residual_energy",
            fields.actual_residual_energy,
            phase_scalar,
        ),
        (
            "actual_common_hidden",
            fields.actual_common_hidden,
            phase_hidden,
        ),
        (
            "actual_common_energy",
            fields.actual_common_energy,
            phase_scalar,
        ),
        (
            "center_phase_weight",
            fields.center_phase_weight,
            (phases, hidden),
        ),
        ("flip_delta", fields.flip_delta, phase_hidden),
        (
            "flipped_center_phase_value",
            fields.flipped_center_phase_value,
            (batch, phases, height, width),
        ),
        (
            "flipped_occupancy_affine",
            fields.flipped_occupancy_affine,
            phase_hidden,
        ),
        (
            "flipped_occupancy_only_joint_affine",
            fields.flipped_occupancy_only_joint_affine,
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
            "flipped_common_silu",
            fields.flipped_common_silu,
            phase_hidden,
        ),
        (
            "flipped_residual_hidden",
            fields.flipped_residual_hidden,
            phase_hidden,
        ),
        (
            "flipped_residual_energy",
            fields.flipped_residual_energy,
            phase_scalar,
        ),
        (
            "flipped_common_hidden",
            fields.flipped_common_hidden,
            phase_hidden,
        ),
        (
            "flipped_common_energy",
            fields.flipped_common_energy,
            phase_scalar,
        ),
        (
            "residual_odd_interaction",
            fields.residual_odd_interaction,
            phase_scalar,
        ),
        (
            "common_even_energy",
            fields.common_even_energy,
            phase_scalar,
        ),
        ("common_gate", fields.common_gate, phase_scalar),
        (
            "common_gate_zero_saturation",
            fields.common_gate_zero_saturation,
            phase_scalar,
        ),
        (
            "common_gate_two_saturation",
            fields.common_gate_two_saturation,
            phase_scalar,
        ),
        (
            "gated_interaction",
            fields.gated_interaction,
            phase_scalar,
        ),
        (
            "native_phase_field",
            fields.native_phase_field,
            phase_scalar,
        ),
        ("field", fields.field, tuple(occupancy.shape)),
    )
    boolean_names = {
        "phase_occupancy",
        "flipped_center_phase_value",
        "common_gate_zero_saturation",
        "common_gate_two_saturation",
    }
    finite_checks: list[Tensor] = []
    for name, value, shape in expected_shapes:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} has an invalid shape")
        if value.device != feature.device:
            raise ValueError(f"{name} device differs from feature")
        expected_dtype = (
            torch.bool if name in boolean_names else torch.float32
        )
        if value.dtype != expected_dtype:
            raise TypeError(f"{name} has an invalid dtype")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if name not in boolean_names:
            finite_checks.append(torch.isfinite(value).all())
    if fields.output_size != output_size:
        raise ValueError("output_size differs from occupancy")
    with torch.no_grad():
        expected_center_phase_weight = model.joint_state_weight[
            :,
            model.config.feature_channels :,
            model.config.coarse_radius,
            model.config.coarse_radius,
        ].transpose(0, 1)
        expected_flipped_center = ~fields.phase_occupancy
        expected_flip_delta = (
            fields.flipped_center_phase_value.to(
                dtype=torch.float32
            )
            - fields.phase_occupancy.to(dtype=torch.float32)
        ).unsqueeze(2) * fields.center_phase_weight[
            None, :, :, None, None
        ]
        expected_flipped_affine = (
            fields.occupancy_affine.unsqueeze(1)
            + fields.flip_delta
        )
        expected_residual_odd = 0.5 * (
            fields.actual_residual_energy
            - fields.flipped_residual_energy
        )
        expected_common_even = 0.5 * (
            fields.actual_common_energy
            + fields.flipped_common_energy
        )
        expected_gate = 2.0 * torch.sigmoid(
            fields.common_even_energy
        )
        expected_zero, expected_two = _gate_saturation_masks(
            fields.common_gate
        )
        expected_interaction = (
            fields.common_gate * fields.residual_odd_interaction
        )
        expected_native_field = (
            model.config.field_amplitude
            + fields.gated_interaction
        )
        expected_field = F.pixel_shuffle(
            fields.native_phase_field,
            model.config.feature_stride,
        )
        equation_checks = (
            fields.center_phase_weight.eq(
                expected_center_phase_weight
            ).all(),
            fields.flipped_center_phase_value.eq(
                expected_flipped_center
            ).all(),
            fields.flip_delta.eq(expected_flip_delta).all(),
            fields.flipped_occupancy_affine.eq(
                expected_flipped_affine
            ).all(),
            fields.residual_odd_interaction.eq(
                expected_residual_odd
            ).all(),
            fields.common_even_energy.eq(
                expected_common_even
            ).all(),
            fields.common_gate.eq(expected_gate).all(),
            fields.common_gate_zero_saturation.eq(
                expected_zero
            ).all(),
            fields.common_gate_two_saturation.eq(
                expected_two
            ).all(),
            fields.gated_interaction.eq(
                expected_interaction
            ).all(),
            fields.native_phase_field.eq(
                expected_native_field
            ).all(),
            fields.field.eq(expected_field).all(),
            (
                (fields.common_gate >= 0.0)
                & (fields.common_gate <= 2.0)
            ).all(),
        )
    finite_checks.extend(equation_checks)
    if not bool(torch.stack(finite_checks).all()):
        raise FloatingPointError(
            "GCR-PACRE finite/equation/gate contract failed"
        )


def validate_gcr_pacre_fields_contract(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    fields: CoverageStateGCRPACREFields,
    *,
    feature: Tensor,
    occupancy: Tensor,
) -> None:
    """Validate externally supplied fields against the v24 machine contract."""

    _validate_gcr_pacre_fields_contract(
        model,
        fields,
        feature=feature,
        occupancy=occupancy,
        inputs_already_validated=False,
    )


def validate_gcr_pacre_fields(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    fields: CoverageStateGCRPACREFields,
    *,
    feature: Tensor,
    occupancy: Tensor,
) -> None:
    """Fail closed unless all v24 audit fields replay from inputs and weights."""

    validate_gcr_pacre_fields_contract(
        model,
        fields,
        feature=feature,
        occupancy=occupancy,
    )
    # Full input-bound replay is intentionally no-grad: validation must not
    # add a graph or mutate model gradients.  It is an explicit verifier and
    # is deliberately not called by the 32,000-update training forward.
    with torch.no_grad():
        encoded = normalize_cslf_feature(
            feature,
            epsilon=model.config.normalization_epsilon,
        )
        phase = pixel_unshuffle_bool_occupancy(
            occupancy,
            stride=model.config.feature_stride,
        )
        (
            occupancy_affine,
            coarse_feature_affine,
            upsampled_feature_affine,
            phase_feature_affine,
        ) = model._affine_states(encoded, phase)
        phase_mean, phase_residual = (
            _phase_centered_feature_affine(
                phase_feature_affine
            )
        )
        actual = model._compatibility_components(
            occupancy_affine.unsqueeze(1),
            phase_feature_affine,
            phase_mean,
        )
        center = model.config.coarse_radius
        center_phase_weight = model.occupancy_weight[
            :, :, center, center
        ].transpose(0, 1).contiguous()
        flipped_center_phase_value = (~phase).contiguous()
        flip_delta = (
            flipped_center_phase_value.to(dtype=encoded.dtype)
            - phase.to(dtype=encoded.dtype)
        ).unsqueeze(2) * center_phase_weight[
            None, :, :, None, None
        ]
        flipped_occupancy_affine = (
            occupancy_affine.unsqueeze(1) + flip_delta
        )
        flipped = model._compatibility_components(
            flipped_occupancy_affine,
            phase_feature_affine,
            phase_mean,
        )
        residual_odd = 0.5 * (actual[5] - flipped[5])
        common_even = 0.5 * (actual[7] + flipped[7])
        expected_gate = 2.0 * torch.sigmoid(common_even)
        zero, two = _gate_saturation_masks(expected_gate)
        interaction = expected_gate * residual_odd
        native_field = model.config.field_amplitude + interaction
        output_field = model.pixel_shuffle(native_field)
        replay = (
            ("encoded_feature", fields.encoded_feature, encoded),
            ("phase_occupancy", fields.phase_occupancy, phase),
            (
                "occupancy_affine",
                fields.occupancy_affine,
                occupancy_affine,
            ),
            (
                "coarse_feature_affine",
                fields.coarse_feature_affine,
                coarse_feature_affine,
            ),
            (
                "upsampled_feature_affine",
                fields.upsampled_feature_affine,
                upsampled_feature_affine,
            ),
            (
                "phase_feature_affine",
                fields.phase_feature_affine,
                phase_feature_affine,
            ),
            (
                "phase_feature_mean",
                fields.phase_feature_mean,
                phase_mean,
            ),
            (
                "phase_feature_residual",
                fields.phase_feature_residual,
                phase_residual,
            ),
            (
                "actual_occupancy_only_joint_affine",
                fields.actual_occupancy_only_joint_affine,
                actual[0],
            ),
            (
                "actual_common_joint_affine",
                fields.actual_common_joint_affine,
                actual[1],
            ),
            (
                "actual_specific_joint_affine",
                fields.actual_specific_joint_affine,
                actual[2],
            ),
            (
                "actual_common_silu",
                fields.actual_common_silu,
                actual[3],
            ),
            (
                "actual_residual_hidden",
                fields.actual_residual_hidden,
                actual[4],
            ),
            (
                "actual_residual_energy",
                fields.actual_residual_energy,
                actual[5],
            ),
            (
                "actual_common_hidden",
                fields.actual_common_hidden,
                actual[6],
            ),
            (
                "actual_common_energy",
                fields.actual_common_energy,
                actual[7],
            ),
            (
                "center_phase_weight",
                fields.center_phase_weight,
                center_phase_weight,
            ),
            ("flip_delta", fields.flip_delta, flip_delta),
            (
                "flipped_center_phase_value",
                fields.flipped_center_phase_value,
                flipped_center_phase_value,
            ),
            (
                "flipped_occupancy_affine",
                fields.flipped_occupancy_affine,
                flipped_occupancy_affine,
            ),
            (
                "flipped_occupancy_only_joint_affine",
                fields.flipped_occupancy_only_joint_affine,
                flipped[0],
            ),
            (
                "flipped_common_joint_affine",
                fields.flipped_common_joint_affine,
                flipped[1],
            ),
            (
                "flipped_specific_joint_affine",
                fields.flipped_specific_joint_affine,
                flipped[2],
            ),
            (
                "flipped_common_silu",
                fields.flipped_common_silu,
                flipped[3],
            ),
            (
                "flipped_residual_hidden",
                fields.flipped_residual_hidden,
                flipped[4],
            ),
            (
                "flipped_residual_energy",
                fields.flipped_residual_energy,
                flipped[5],
            ),
            (
                "flipped_common_hidden",
                fields.flipped_common_hidden,
                flipped[6],
            ),
            (
                "flipped_common_energy",
                fields.flipped_common_energy,
                flipped[7],
            ),
            (
                "residual_odd_interaction",
                fields.residual_odd_interaction,
                residual_odd,
            ),
            (
                "common_even_energy",
                fields.common_even_energy,
                common_even,
            ),
            ("common_gate", fields.common_gate, expected_gate),
            (
                "common_gate_zero_saturation",
                fields.common_gate_zero_saturation,
                zero,
            ),
            (
                "common_gate_two_saturation",
                fields.common_gate_two_saturation,
                two,
            ),
            (
                "gated_interaction",
                fields.gated_interaction,
                interaction,
            ),
            (
                "native_phase_field",
                fields.native_phase_field,
                native_field,
            ),
            ("field", fields.field, output_field),
        )
        for name, actual_value, expected_value in replay:
            _require_exact_tensor(
                name=name,
                actual=actual_value,
                expected=expected_value.contiguous(),
            )
        reconstructed = row_major_phase_unpack(
            fields.phase_feature_affine,
            stride=model.config.feature_stride,
        )
        _require_exact_tensor(
            name="phase_transport_roundtrip",
            actual=fields.upsampled_feature_affine,
            expected=reconstructed,
        )


CURELiteGatedCommonResidualPACREField = (
    CURELiteGatedCommonResidualPACRELevelSet
)


__all__ = [
    "CSLF_GCR_PACRE_EQUATION_POLICY",
    "CSLF_GCR_PACRE_FIELD_POLICY",
    "CURELiteGatedCommonResidualPACREField",
    "CURELiteGatedCommonResidualPACRELevelSet",
    "CoverageStateGCRPACREConfig",
    "CoverageStateGCRPACREFields",
    "GCRPACREFP64Comparison",
    "GCRPACREFP64OracleFields",
    "GCR_PACRE_CANDIDATE",
    "GCR_PACRE_CENTERING_POLICY",
    "GCR_PACRE_ENERGY_POLICY",
    "GCR_PACRE_FIELDS_FQCN",
    "GCR_PACRE_FP64_ORACLE_ABS_TOL",
    "GCR_PACRE_FP64_ORACLE_MAX_ULP",
    "GCR_PACRE_GATE_STATISTICS_SCHEMA",
    "GCR_PACRE_INTERACTION_POLICY",
    "GCR_PACRE_METHOD_ID",
    "GCR_PACRE_NUMERICAL_POLICY",
    "GCRPACREGateSaturationAudit",
    "compare_gcr_pacre_fp32_to_fp64_oracle",
    "gcr_pacre_fp32_ulp_distance",
    "summarize_gcr_pacre_gate_saturation",
    "validate_gcr_pacre_fields",
    "validate_gcr_pacre_fields_contract",
]
