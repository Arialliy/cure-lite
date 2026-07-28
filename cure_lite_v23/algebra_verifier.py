"""Numerically honest algebra verification for PACRE-VC v23.

PACRE-VC intentionally inherits the PACRE-v22 floating-point forward.  This
module therefore verifies the operations that the forward actually performs:

* same-device exact replay for deterministic algebraic FP32 operations;
* a frozen operand-scaled bound for repeated SiLU evaluation; and
* CPU-FP64 evaluation of conservative FP32 phase-semantic bounds.

No check in this module changes model state, retains gradients, or reads data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor
from torch.nn import functional as F

from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREFields,
)


PACRE_VC_ALGEBRA_POLICY: Final = (
    "v22_same_device_exact_forward_replay_"
    "bounded_repeated_silu_difference_"
    "cpu_float64_ftz_safe_phase_bounds_v1"
)
EPS32: Final = float(torch.finfo(torch.float32).eps)
EPS64: Final = float(torch.finfo(torch.float64).eps)
# ``tiny`` is the smallest *normal* FP32 number.  Unlike a minimum-subnormal
# floor, this remains conservative when a CPU or accelerator flushes FP32
# subnormals to zero.
TINY32: Final = float(torch.finfo(torch.float32).tiny)
# Replaying SiLU over a contiguous stored tensor need not use the same CPU
# vector kernel as the original forward over an expanded view.  Four machine
# epsilons cover two repeated SiLU evaluations and their two rounded
# subtractions without turning this diagnostic into a broad allclose.
PACRE_VC_SILU_REPLAY_FACTOR: Final = 4.0


def gamma(operation_count: int, epsilon: float) -> float:
    """Return the standard conservative ``gamma_n`` rounding factor."""

    if (
        isinstance(operation_count, bool)
        or not isinstance(operation_count, int)
        or operation_count < 1
        or isinstance(epsilon, bool)
        or not isinstance(epsilon, float)
        or not 0.0 < epsilon < 1.0
    ):
        raise ValueError("gamma arguments are invalid")
    product = float(operation_count) * epsilon
    if product >= 1.0:
        raise ValueError("gamma bound is undefined")
    return product / (1.0 - product)


@dataclass(frozen=True)
class PACREVCSubcheck:
    """One deterministic, receipt-ready numerical observation."""

    name: str
    gate_eligible: bool
    passed: bool
    failed_element_count: int
    maximum_error: float
    maximum_bound: float
    bound_at_maximum_error: float
    argmax_coordinate: tuple[int, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "gate_eligible": self.gate_eligible,
            "passed": self.passed,
            "failed_element_count": self.failed_element_count,
            "maximum_error_hex": self.maximum_error.hex(),
            "maximum_bound_hex": self.maximum_bound.hex(),
            "bound_at_maximum_error_hex": (
                self.bound_at_maximum_error.hex()
            ),
            "argmax_coordinate": list(self.argmax_coordinate),
        }


@dataclass(frozen=True)
class PACREVCExactReplayReport:
    """PACRE replay: algebraic checks exact, repeated SiLU checks bounded.

    The historical class/field name is retained because generated-stress and
    receipt code already consume this API.  The payload explicitly identifies
    the two non-exact transcendental checks.
    """

    checks: tuple[PACREVCSubcheck, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def canonical_payload(self) -> dict[str, object]:
        bounded = [
            check.name
            for check in self.checks
            if check.name.endswith("_bounded")
        ]
        return {
            "policy": PACRE_VC_ALGEBRA_POLICY,
            "algebraic_replay_exact": True,
            "transcendental_replay_bounded": True,
            "silu_replay_factor_hex": (
                PACRE_VC_SILU_REPLAY_FACTOR.hex()
            ),
            "bounded_transcendental_checks": bounded,
            "passed": self.passed,
            "checks": [
                check.canonical_payload() for check in self.checks
            ],
        }


@dataclass(frozen=True)
class PACREVCPhaseSemanticReport:
    """Analytic reconstruction and centering observations."""

    reconstruction: PACREVCSubcheck
    centering: PACREVCSubcheck

    @property
    def passed(self) -> bool:
        return self.reconstruction.passed and self.centering.passed

    def canonical_payload(self) -> dict[str, object]:
        return {
            "policy": PACRE_VC_ALGEBRA_POLICY,
            "epsilon32_hex": EPS32.hex(),
            "epsilon64_hex": EPS64.hex(),
            "tiny32_hex": TINY32.hex(),
            "ftz_safe_floor": True,
            "passed": self.passed,
            "reconstruction": self.reconstruction.canonical_payload(),
            "centering": self.centering.canonical_payload(),
        }


@dataclass(frozen=True)
class PACREVCAlgebraVerification:
    """Complete gate-eligible PACRE-VC algebra verification."""

    exact_replay: PACREVCExactReplayReport
    phase_semantics: PACREVCPhaseSemanticReport

    @property
    def passed(self) -> bool:
        return self.exact_replay.passed and self.phase_semantics.passed

    def canonical_payload(self) -> dict[str, object]:
        return {
            "policy": PACRE_VC_ALGEBRA_POLICY,
            "passed": self.passed,
            "exact_replay": self.exact_replay.canonical_payload(),
            "phase_semantics": self.phase_semantics.canonical_payload(),
        }


def _first_argmax_coordinate(value: Tensor) -> tuple[int, ...]:
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.numel() < 1
        or not value.is_contiguous()
    ):
        raise ValueError("argmax tensor must be nonempty contiguous CPU FP64")
    flat_index = int(torch.argmax(value.reshape(-1)).item())
    coordinate: list[int] = []
    remainder = flat_index
    for extent in reversed(value.shape):
        coordinate.append(remainder % int(extent))
        remainder //= int(extent)
    return tuple(reversed(coordinate))


def _observation(
    name: str,
    error: Tensor,
    bound: Tensor,
    *,
    failed: Tensor | None = None,
) -> PACREVCSubcheck:
    if (
        error.device.type != "cpu"
        or bound.device.type != "cpu"
        or error.dtype != torch.float64
        or bound.dtype != torch.float64
        or error.shape != bound.shape
        or error.numel() < 1
        or not error.is_contiguous()
        or not bound.is_contiguous()
        or not bool(torch.isfinite(error).all())
        or not bool(torch.isfinite(bound).all())
        or bool(torch.any(error < 0.0))
        or bool(torch.any(bound < 0.0))
    ):
        raise ValueError("numerical observation tensors are invalid")
    if failed is None:
        failed = error > bound
    if (
        failed.device.type != "cpu"
        or failed.dtype != torch.bool
        or failed.shape != error.shape
    ):
        raise ValueError("failed mask differs from numerical observation")
    coordinate = _first_argmax_coordinate(error)
    flat_index = 0
    stride = 1
    for extent, index in zip(
        reversed(error.shape),
        reversed(coordinate),
        strict=True,
    ):
        flat_index += index * stride
        stride *= int(extent)
    flat_error = error.reshape(-1)
    flat_bound = bound.reshape(-1)
    maximum_error = float(flat_error[flat_index].item())
    return PACREVCSubcheck(
        name=name,
        gate_eligible=True,
        passed=not bool(torch.any(failed)),
        failed_element_count=int(failed.sum().item()),
        maximum_error=maximum_error,
        maximum_bound=float(torch.max(bound).item()),
        bound_at_maximum_error=float(flat_bound[flat_index].item()),
        argmax_coordinate=coordinate,
    )


def _require_tensor(
    value: Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    finite: bool,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape differs from the PACRE contract")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype differs from the PACRE contract")
    if value.device != device:
        raise ValueError(f"{name} device differs from the PACRE contract")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if finite and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} must be finite")


def validate_pacre_fields_contract(
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    fields: CoverageStatePACREFields,
) -> None:
    """Validate every tensor used by the replay before arithmetic begins."""

    if not isinstance(
        model,
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    ):
        raise TypeError("model must inherit the PACRE-v22 model")
    if type(fields) is not CoverageStatePACREFields:
        raise TypeError("fields must have exact CoverageStatePACREFields type")

    phase = fields.phase_feature_affine
    if not isinstance(phase, Tensor) or phase.ndim != 5:
        raise ValueError("phase_feature_affine must be five-dimensional")
    batch, phases, width, height, columns = (
        int(value) for value in phase.shape
    )
    if min(batch, phases, width, height, columns) < 1:
        raise ValueError("PACRE fields cannot have an empty dimension")
    expected_phases = int(model.config.feature_stride) ** 2
    if (
        phases != expected_phases
        or width != int(model.config.width)
    ):
        raise ValueError("phase dimensions differ from model config")

    device = phase.device
    phase_hidden = (batch, phases, width, height, columns)
    native = (batch, phases, height, columns)
    output_height = height * int(model.config.feature_stride)
    output_width = columns * int(model.config.feature_stride)
    output = (batch, 1, output_height, output_width)
    float_shapes = {
        "encoded_feature": (
            fields.encoded_feature,
            (
                batch,
                int(model.config.feature_channels),
                height,
                columns,
            ),
        ),
        "occupancy_affine": (
            fields.occupancy_affine,
            (batch, width, height, columns),
        ),
        "coarse_feature_affine": (
            fields.coarse_feature_affine,
            (batch, width, height, columns),
        ),
        "upsampled_feature_affine": (
            fields.upsampled_feature_affine,
            (batch, width, output_height, output_width),
        ),
        "phase_feature_affine": (phase, phase_hidden),
        "phase_feature_mean": (
            fields.phase_feature_mean,
            (batch, 1, width, height, columns),
        ),
        "phase_feature_residual": (
            fields.phase_feature_residual,
            phase_hidden,
        ),
        "actual_common_joint_affine": (
            fields.actual_common_joint_affine,
            phase_hidden,
        ),
        "actual_specific_joint_affine": (
            fields.actual_specific_joint_affine,
            phase_hidden,
        ),
        "actual_compatibility_hidden": (
            fields.actual_compatibility_hidden,
            phase_hidden,
        ),
        "actual_compatibility_energy": (
            fields.actual_compatibility_energy,
            native,
        ),
        "flip_delta": (fields.flip_delta, phase_hidden),
        "flipped_occupancy_affine": (
            fields.flipped_occupancy_affine,
            phase_hidden,
        ),
        "flipped_common_joint_affine": (
            fields.flipped_common_joint_affine,
            phase_hidden,
        ),
        "flipped_specific_joint_affine": (
            fields.flipped_specific_joint_affine,
            phase_hidden,
        ),
        "flipped_compatibility_hidden": (
            fields.flipped_compatibility_hidden,
            phase_hidden,
        ),
        "flipped_compatibility_energy": (
            fields.flipped_compatibility_energy,
            native,
        ),
        "native_phase_interaction": (
            fields.native_phase_interaction,
            native,
        ),
        "native_phase_field": (fields.native_phase_field, native),
        "field": (fields.field, output),
    }
    for name, (value, shape) in float_shapes.items():
        _require_tensor(
            value,
            name=name,
            shape=shape,
            dtype=torch.float32,
            device=device,
            finite=True,
        )
    _require_tensor(
        fields.phase_occupancy,
        name="phase_occupancy",
        shape=(batch, phases, height, columns),
        dtype=torch.bool,
        device=device,
        finite=False,
    )
    if fields.output_size != (output_height, output_width):
        raise ValueError("output_size differs from PACRE field shape")

    for name, parameter, shape in (
        (
            "scalar_energy_weight",
            model.scalar_energy_weight,
            (width,),
        ),
        (
            "joint_hidden_bias",
            model.joint_hidden_bias,
            (width,),
        ),
    ):
        _require_tensor(
            parameter,
            name=name,
            shape=shape,
            dtype=torch.float32,
            device=device,
            finite=True,
        )
    if (
        model.joint_state_weight.dtype != torch.float32
        or model.joint_state_weight.device != device
        or not model.joint_state_weight.is_contiguous()
        or not bool(torch.isfinite(model.joint_state_weight).all())
    ):
        raise ValueError("joint_state_weight differs from PACRE contract")


def _require_same_contract(
    actual: Tensor,
    expected: Tensor,
    *,
    name: str,
) -> None:
    if not isinstance(actual, Tensor) or not isinstance(expected, Tensor):
        raise TypeError(f"{name} replay values must be tensors")
    if actual.shape != expected.shape:
        raise ValueError(f"{name} replay shape differs")
    if actual.dtype != expected.dtype:
        raise TypeError(f"{name} replay dtype differs")
    if actual.device != expected.device:
        raise ValueError(f"{name} replay device differs")
    if not actual.is_contiguous() or not expected.is_contiguous():
        raise ValueError(f"{name} replay tensors must be contiguous")
    if (
        not bool(torch.isfinite(actual).all())
        or not bool(torch.isfinite(expected).all())
    ):
        raise FloatingPointError(f"{name} replay tensors must be finite")


def _exact_replay(
    name: str,
    actual: Tensor,
    expected: Tensor,
) -> PACREVCSubcheck:
    _require_same_contract(actual, expected, name=name)
    unequal = actual.ne(expected).detach().to("cpu").contiguous()
    error = (
        actual.detach().to(device="cpu", dtype=torch.float64)
        - expected.detach().to(device="cpu", dtype=torch.float64)
    ).abs().contiguous()
    bound = torch.zeros_like(error)
    return _observation(name, error, bound, failed=unequal)


def _bounded_silu_difference_replay(
    name: str,
    actual: Tensor,
    specific: Tensor,
    common: Tensor,
) -> PACREVCSubcheck:
    """Replay one SiLU difference with a frozen FP32 scale bound.

    PyTorch may select different vector kernels for the original expanded
    common tensor and its later contiguous stored copy.  Consequently repeated
    transcendental evaluation is not a raw-bit identity even though the input
    values are identical.  This bound is deliberately local to the two replay
    operands and remains orders of magnitude below a material field change.
    """

    specific_silu = F.silu(specific).contiguous()
    common_silu = F.silu(common).contiguous()
    expected = (specific_silu - common_silu).contiguous()
    _require_same_contract(actual, expected, name=name)
    actual64 = actual.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    expected64 = expected.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    specific_silu64 = specific_silu.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    common_silu64 = common_silu.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    error = (actual64 - expected64).abs().contiguous()
    bound = (
        PACRE_VC_SILU_REPLAY_FACTOR
        * EPS32
        * (
            1.0
            + specific_silu64.abs()
            + common_silu64.abs()
        )
        + TINY32
    ).contiguous()
    return _observation(name, error, bound)


@torch.no_grad()
def verify_exact_forward_replay(
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    fields: CoverageStatePACREFields,
) -> PACREVCExactReplayReport:
    """Replay the complete PACRE downstream algebra without mutation."""

    validate_pacre_fields_contract(model, fields)
    expected_mean = (
        fields.phase_feature_affine.mean(dim=1, keepdim=True).contiguous()
    )
    expected_residual = (
        fields.phase_feature_affine - fields.phase_feature_mean
    ).contiguous()

    center = int(model.config.coarse_radius)
    center_phase_weight = model.occupancy_weight[
        :, :, center, center
    ].transpose(0, 1)
    expected_flip_delta = (
        (
            1.0
            - 2.0
            * fields.phase_occupancy.to(
                dtype=fields.encoded_feature.dtype
            )
        ).unsqueeze(2)
        * center_phase_weight[None, :, :, None, None]
    ).contiguous()
    expected_flipped_occupancy = (
        fields.occupancy_affine.unsqueeze(1) + fields.flip_delta
    ).contiguous()

    expected_actual_common = (
        (
            fields.occupancy_affine.unsqueeze(1)
            + fields.phase_feature_mean
        )
        .expand_as(fields.phase_feature_affine)
        .contiguous()
    )
    expected_actual_specific = (
        fields.occupancy_affine.unsqueeze(1)
        + fields.phase_feature_affine
    ).contiguous()
    expected_flipped_common = (
        fields.flipped_occupancy_affine + fields.phase_feature_mean
    ).contiguous()
    expected_flipped_specific = (
        fields.flipped_occupancy_affine + fields.phase_feature_affine
    ).contiguous()
    readout = model.scalar_energy_weight[None, None, :, None, None]
    expected_actual_energy = (
        fields.actual_compatibility_hidden * readout
    ).sum(dim=2).contiguous()
    expected_flipped_energy = (
        fields.flipped_compatibility_hidden * readout
    ).sum(dim=2).contiguous()
    expected_interaction = (
        0.5
        * (
            fields.actual_compatibility_energy
            - fields.flipped_compatibility_energy
        )
    ).contiguous()
    expected_native_field = (
        model.config.field_amplitude + fields.native_phase_interaction
    ).contiguous()
    expected_field = model.pixel_shuffle(
        fields.native_phase_field
    ).contiguous()

    checks = (
        _exact_replay(
            "phase_mean_forward_exact",
            fields.phase_feature_mean,
            expected_mean,
        ),
        _exact_replay(
            "phase_residual_forward_exact",
            fields.phase_feature_residual,
            expected_residual,
        ),
        _exact_replay(
            "flip_delta_forward_exact",
            fields.flip_delta,
            expected_flip_delta,
        ),
        _exact_replay(
            "flipped_occupancy_affine_forward_exact",
            fields.flipped_occupancy_affine,
            expected_flipped_occupancy,
        ),
        _exact_replay(
            "actual_common_forward_exact",
            fields.actual_common_joint_affine,
            expected_actual_common,
        ),
        _exact_replay(
            "actual_specific_forward_exact",
            fields.actual_specific_joint_affine,
            expected_actual_specific,
        ),
        _exact_replay(
            "flipped_common_forward_exact",
            fields.flipped_common_joint_affine,
            expected_flipped_common,
        ),
        _exact_replay(
            "flipped_specific_forward_exact",
            fields.flipped_specific_joint_affine,
            expected_flipped_specific,
        ),
        _bounded_silu_difference_replay(
            "actual_hidden_forward_bounded",
            fields.actual_compatibility_hidden,
            fields.actual_specific_joint_affine,
            fields.actual_common_joint_affine,
        ),
        _bounded_silu_difference_replay(
            "flipped_hidden_forward_bounded",
            fields.flipped_compatibility_hidden,
            fields.flipped_specific_joint_affine,
            fields.flipped_common_joint_affine,
        ),
        _exact_replay(
            "actual_energy_forward_exact",
            fields.actual_compatibility_energy,
            expected_actual_energy,
        ),
        _exact_replay(
            "flipped_energy_forward_exact",
            fields.flipped_compatibility_energy,
            expected_flipped_energy,
        ),
        _exact_replay(
            "native_interaction_forward_exact",
            fields.native_phase_interaction,
            expected_interaction,
        ),
        _exact_replay(
            "native_field_forward_exact",
            fields.native_phase_field,
            expected_native_field,
        ),
        _exact_replay(
            "output_field_forward_exact",
            fields.field,
            expected_field,
        ),
    )
    return PACREVCExactReplayReport(checks=checks)


def _require_phase_contract(
    phase_affine: Tensor,
    phase_mean: Tensor,
    phase_residual: Tensor,
) -> None:
    tensors = (phase_affine, phase_mean, phase_residual)
    if any(not isinstance(value, Tensor) for value in tensors):
        raise TypeError("phase bound inputs must be tensors")
    if (
        phase_affine.ndim != 5
        or phase_residual.shape != phase_affine.shape
        or phase_mean.shape
        != (
            phase_affine.shape[0],
            1,
            *phase_affine.shape[2:],
        )
        or min(phase_affine.shape) < 1
    ):
        raise ValueError("phase bound shapes differ")
    for name, value in (
        ("phase_affine", phase_affine),
        ("phase_mean", phase_mean),
        ("phase_residual", phase_residual),
    ):
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be FP32")
        if value.device != phase_affine.device:
            raise ValueError(f"{name} device differs")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} must be finite")


@torch.no_grad()
def phase_roundoff_observations(
    phase_affine: Tensor,
    phase_mean: Tensor,
    phase_residual: Tensor,
) -> PACREVCPhaseSemanticReport:
    """Evaluate the frozen FTZ-safe phase bounds in canonical CPU FP64."""

    _require_phase_contract(
        phase_affine,
        phase_mean,
        phase_residual,
    )
    x = phase_affine.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    m = phase_mean.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    r = phase_residual.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    phases = int(x.shape[1])
    if phases < 2:
        raise ValueError("phase centering requires at least two phases")
    expanded_mean = m.expand_as(x)
    reconstruction_error = (
        expanded_mean + r - x
    ).abs().contiguous()
    reconstruction_bound = (
        EPS32 * (x.abs() + expanded_mean.abs())
        + 2.0 * TINY32
    ).contiguous()

    center_error = r.sum(dim=1).abs().contiguous()
    sx = x.abs().sum(dim=1)
    sm = float(phases) * m[:, 0].abs()
    sr = r.abs().sum(dim=1)
    center_bound = (
        gamma(phases, EPS32) * sx
        + EPS32 * (sx + sm)
        + gamma(phases - 1, EPS64) * sr
        + float(3 * phases + 1) * TINY32
    ).contiguous()

    return PACREVCPhaseSemanticReport(
        reconstruction=_observation(
            "phase_reconstruction_roundoff_bound",
            reconstruction_error,
            reconstruction_bound,
        ),
        centering=_observation(
            "phase_centering_roundoff_bound",
            center_error,
            center_bound,
        ),
    )


@torch.no_grad()
def verify_phase_semantics(
    fields: CoverageStatePACREFields,
) -> PACREVCPhaseSemanticReport:
    if type(fields) is not CoverageStatePACREFields:
        raise TypeError("fields must have exact CoverageStatePACREFields type")
    return phase_roundoff_observations(
        fields.phase_feature_affine,
        fields.phase_feature_mean,
        fields.phase_feature_residual,
    )


@torch.no_grad()
def verify_pacre_v22_forward_fields(
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    fields: CoverageStatePACREFields,
) -> PACREVCAlgebraVerification:
    """Run every gate-eligible PACRE-VC algebra check."""

    exact = verify_exact_forward_replay(model, fields)
    phase = verify_phase_semantics(fields)
    return PACREVCAlgebraVerification(
        exact_replay=exact,
        phase_semantics=phase,
    )


verify_pacre_forward_fields = verify_pacre_v22_forward_fields


__all__ = [
    "EPS32",
    "EPS64",
    "PACRE_VC_ALGEBRA_POLICY",
    "PACRE_VC_SILU_REPLAY_FACTOR",
    "TINY32",
    "PACREVCAlgebraVerification",
    "PACREVCExactReplayReport",
    "PACREVCPhaseSemanticReport",
    "PACREVCSubcheck",
    "gamma",
    "phase_roundoff_observations",
    "validate_pacre_fields_contract",
    "verify_exact_forward_replay",
    "verify_pacre_forward_fields",
    "verify_pacre_v22_forward_fields",
    "verify_phase_semantics",
]
