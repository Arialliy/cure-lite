"""Diagnostic-only numerical evidence for PACRE-VC v23.

The functions here never authorize a model by error magnitude.  They provide
one frozen definition for normalized error, reference-relative error, FP32 ULP
distance, legacy subtraction mismatch, a local CPU-FP64 compatibility
reference, and complete signal swallowing.  Oracle integrity is reported
separately from all numerical magnitudes.
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

from .algebra_verifier import TINY32, validate_pacre_fields_contract


PACRE_VC_DIAGNOSTIC_POLICY: Final = (
    "diagnostic_only_unique_error_ulp_and_local_fp64_reference_v1"
)
PACRE_VC_NORMALIZED_ERROR_POLICY: Final = (
    "abs_actual_minus_reference_over_"
    "one_plus_abs_actual_plus_abs_reference_v1"
)
PACRE_VC_RELATIVE_ERROR_POLICY: Final = (
    "abs_actual_minus_reference_over_"
    "max_abs_reference_and_fp32_tiny_v1"
)
PACRE_VC_ULP_POLICY: Final = (
    "finite_fp32_monotone_ordered_bits_signed_zero_coalesced_v1"
)
PACRE_VC_FP64_ORACLE_POLICY: Final = (
    "stored_fp32_operands_cpu_float64_downstream_reference_v1"
)
PACRE_VC_FIXED_READOUT_POLICY: Final = (
    "cpu_fp32_linspace_0.5_to_1.5_width_then_copy_v1"
)
PACRE_VC_LEGACY_RTOL: Final = 2.0e-6
PACRE_VC_LEGACY_ATOL: Final = 2.0e-7


def _first_argmax_coordinate(value: Tensor) -> tuple[int, ...]:
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.numel() < 1
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("argmax tensor must be finite contiguous CPU FP64")
    flat_index = int(torch.argmax(value.reshape(-1)).item())
    coordinate: list[int] = []
    remainder = flat_index
    for extent in reversed(value.shape):
        coordinate.append(remainder % int(extent))
        remainder //= int(extent)
    return tuple(reversed(coordinate))


def _value_at(value: Tensor, coordinate: tuple[int, ...]) -> float:
    return float(value[coordinate].item())


def _aligned_cpu_float64(
    actual: Tensor,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    if (
        not isinstance(actual, Tensor)
        or not isinstance(reference, Tensor)
        or actual.shape != reference.shape
        or actual.numel() < 1
        or not actual.is_floating_point()
        or not reference.is_floating_point()
        or not bool(torch.isfinite(actual).all())
        or not bool(torch.isfinite(reference).all())
    ):
        raise ValueError("diagnostic tensors must be aligned finite floating")
    actual64 = actual.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    reference64 = reference.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    return actual64, reference64


@torch.no_grad()
def normalized_absolute_error(
    actual: Tensor,
    reference: Tensor,
) -> Tensor:
    """Return ``|a-r| / (1 + |a| + |r|)`` in canonical CPU FP64."""

    actual64, reference64 = _aligned_cpu_float64(actual, reference)
    return (
        (actual64 - reference64).abs()
        / (1.0 + actual64.abs() + reference64.abs())
    ).contiguous()


@torch.no_grad()
def reference_relative_error(
    actual: Tensor,
    reference: Tensor,
) -> Tensor:
    """Return ``|a-r| / max(|r|, TINY32)`` in canonical CPU FP64."""

    actual64, reference64 = _aligned_cpu_float64(actual, reference)
    floor = torch.full_like(reference64, TINY32)
    return (
        (actual64 - reference64).abs()
        / torch.maximum(reference64.abs(), floor)
    ).contiguous()


def _ordered_fp32_key(value: Tensor) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.float32
        or value.numel() < 1
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("ULP inputs must be nonempty finite FP32")
    raw = (
        value.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.int32)
        .to(torch.int64)
        & 0xFFFFFFFF
    )
    magnitude = raw & 0x7FFFFFFF
    negative = (raw & 0x80000000) != 0
    # Both signed zeros map to 0x80000000.  All other finite values are
    # monotone in their numerical ordering.
    return torch.where(
        negative,
        0x80000000 - magnitude,
        0x80000000 + magnitude,
    )


@torch.no_grad()
def fp32_ulp_distance(first: Tensor, second: Tensor) -> Tensor:
    """Return the frozen signed-zero-coalesced FP32 ULP distance."""

    if (
        not isinstance(first, Tensor)
        or not isinstance(second, Tensor)
        or first.shape != second.shape
        or first.dtype != torch.float32
        or second.dtype != torch.float32
    ):
        raise ValueError("ULP tensors must be aligned FP32")
    return (
        _ordered_fp32_key(first) - _ordered_fp32_key(second)
    ).abs().contiguous()


@dataclass(frozen=True)
class PACREVCErrorSummary:
    element_count: int
    maximum_absolute_error: float
    maximum_normalized_error: float
    maximum_reference_relative_error: float
    maximum_absolute_coordinate: tuple[int, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "element_count": self.element_count,
            "maximum_absolute_error_hex": (
                self.maximum_absolute_error.hex()
            ),
            "maximum_normalized_error_hex": (
                self.maximum_normalized_error.hex()
            ),
            "maximum_reference_relative_error_hex": (
                self.maximum_reference_relative_error.hex()
            ),
            "maximum_absolute_coordinate": list(
                self.maximum_absolute_coordinate
            ),
            "normalized_error_policy": (
                PACRE_VC_NORMALIZED_ERROR_POLICY
            ),
            "relative_error_policy": PACRE_VC_RELATIVE_ERROR_POLICY,
        }


def _error_summary(
    actual: Tensor,
    reference: Tensor,
) -> PACREVCErrorSummary:
    actual64, reference64 = _aligned_cpu_float64(actual, reference)
    absolute = (actual64 - reference64).abs().contiguous()
    normalized = normalized_absolute_error(actual64, reference64)
    relative = reference_relative_error(actual64, reference64)
    return PACREVCErrorSummary(
        element_count=absolute.numel(),
        maximum_absolute_error=float(torch.max(absolute).item()),
        maximum_normalized_error=float(torch.max(normalized).item()),
        maximum_reference_relative_error=float(
            torch.max(relative).item()
        ),
        maximum_absolute_coordinate=_first_argmax_coordinate(absolute),
    )


@dataclass(frozen=True)
class PACREVCLegacyResidualLane:
    name: str
    gate_eligible: bool
    decision_weight: int
    error: PACREVCErrorSummary
    maximum_ulp_distance: int
    failed_under_v22_allclose_count: int
    common_magnitude_at_argmax: float
    specific_magnitude_at_argmax: float
    residual_magnitude_at_argmax: float

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "gate_eligible": self.gate_eligible,
            "decision_weight": self.decision_weight,
            "error": self.error.canonical_payload(),
            "maximum_ulp_distance": self.maximum_ulp_distance,
            "ulp_policy": PACRE_VC_ULP_POLICY,
            "failed_under_v22_allclose_count": (
                self.failed_under_v22_allclose_count
            ),
            "legacy_rtol_hex": PACRE_VC_LEGACY_RTOL.hex(),
            "legacy_atol_hex": PACRE_VC_LEGACY_ATOL.hex(),
            "common_magnitude_at_argmax_hex": (
                self.common_magnitude_at_argmax.hex()
            ),
            "specific_magnitude_at_argmax_hex": (
                self.specific_magnitude_at_argmax.hex()
            ),
            "residual_magnitude_at_argmax_hex": (
                self.residual_magnitude_at_argmax.hex()
            ),
        }


@dataclass(frozen=True)
class PACREVCLegacyDiagnostics:
    actual: PACREVCLegacyResidualLane
    flipped: PACREVCLegacyResidualLane

    def canonical_payload(self) -> dict[str, object]:
        return {
            "policy": PACRE_VC_DIAGNOSTIC_POLICY,
            "gate_eligible": False,
            "decision_weight": 0,
            "actual": self.actual.canonical_payload(),
            "flipped": self.flipped.canonical_payload(),
        }


def _require_legacy_contract(
    specific: Tensor,
    common: Tensor,
    residual: Tensor,
) -> None:
    if (
        not isinstance(specific, Tensor)
        or not isinstance(common, Tensor)
        or not isinstance(residual, Tensor)
        or specific.shape != common.shape
        or specific.shape != residual.shape
        or specific.numel() < 1
    ):
        raise ValueError("legacy tensors must have one nonempty shape")
    for name, value in (
        ("specific", specific),
        ("common", common),
        ("residual", residual),
    ):
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be FP32")
        if value.device != specific.device:
            raise ValueError(f"{name} device differs")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} must be finite")


@torch.no_grad()
def legacy_residual_lane(
    name: str,
    specific: Tensor,
    common: Tensor,
    residual: Tensor,
) -> PACREVCLegacyResidualLane:
    """Summarize the obsolete subtractive identity without gating on it."""

    if not isinstance(name, str) or not name:
        raise ValueError("legacy lane name must be nonempty")
    _require_legacy_contract(specific, common, residual)
    recovered = (specific - common).contiguous()
    summary = _error_summary(recovered, residual)
    coordinate = summary.maximum_absolute_coordinate
    failed = ~torch.isclose(
        recovered,
        residual,
        rtol=PACRE_VC_LEGACY_RTOL,
        atol=PACRE_VC_LEGACY_ATOL,
    )
    ulp = fp32_ulp_distance(recovered, residual)
    common64 = common.detach().to("cpu", dtype=torch.float64)
    specific64 = specific.detach().to("cpu", dtype=torch.float64)
    residual64 = residual.detach().to("cpu", dtype=torch.float64)
    return PACREVCLegacyResidualLane(
        name=name,
        gate_eligible=False,
        decision_weight=0,
        error=summary,
        maximum_ulp_distance=int(torch.max(ulp).item()),
        failed_under_v22_allclose_count=int(
            failed.sum().detach().cpu().item()
        ),
        common_magnitude_at_argmax=abs(
            _value_at(common64, coordinate)
        ),
        specific_magnitude_at_argmax=abs(
            _value_at(specific64, coordinate)
        ),
        residual_magnitude_at_argmax=abs(
            _value_at(residual64, coordinate)
        ),
    )


@torch.no_grad()
def legacy_subtraction_diagnostics(
    fields: CoverageStatePACREFields,
) -> PACREVCLegacyDiagnostics:
    if type(fields) is not CoverageStatePACREFields:
        raise TypeError("fields must have exact CoverageStatePACREFields type")
    return PACREVCLegacyDiagnostics(
        actual=legacy_residual_lane(
            "actual_legacy_subtractive_residual",
            fields.actual_specific_joint_affine,
            fields.actual_common_joint_affine,
            fields.phase_feature_residual,
        ),
        flipped=legacy_residual_lane(
            "flipped_legacy_subtractive_residual",
            fields.flipped_specific_joint_affine,
            fields.flipped_common_joint_affine,
            fields.phase_feature_residual,
        ),
    )


@dataclass(frozen=True)
class PACREVCCompleteSwallowObservation:
    name: str
    gate_eligible: bool
    eligible_element_count: int
    swallowed_element_count: int
    maximum_swallowed_reference_magnitude: float
    maximum_swallowed_coordinate: tuple[int, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "gate_eligible": self.gate_eligible,
            "definition": (
                "prerequisite_and_reference_nonzero_and_actual_exact_zero"
            ),
            "eligible_element_count": self.eligible_element_count,
            "swallowed_element_count": self.swallowed_element_count,
            "maximum_swallowed_reference_magnitude_hex": (
                self.maximum_swallowed_reference_magnitude.hex()
            ),
            "maximum_swallowed_coordinate": list(
                self.maximum_swallowed_coordinate
            ),
        }


@torch.no_grad()
def complete_swallow_observation(
    name: str,
    actual: Tensor,
    reference: Tensor,
    *,
    prerequisite: Tensor | None = None,
) -> PACREVCCompleteSwallowObservation:
    """Count complete, diagnostic-only loss of a nonzero FP64 reference."""

    if not isinstance(name, str) or not name:
        raise ValueError("swallow name must be nonempty")
    actual64, reference64 = _aligned_cpu_float64(actual, reference)
    if prerequisite is None:
        prerequisite_cpu = torch.ones_like(
            reference64,
            dtype=torch.bool,
        )
    else:
        if (
            not isinstance(prerequisite, Tensor)
            or prerequisite.shape != reference.shape
            or prerequisite.dtype != torch.bool
        ):
            raise ValueError("swallow prerequisite must be aligned bool")
        prerequisite_cpu = prerequisite.detach().to("cpu").contiguous()
    eligible = prerequisite_cpu & reference64.ne(0.0)
    swallowed = eligible & actual64.eq(0.0)
    swallowed_magnitude = torch.where(
        swallowed,
        reference64.abs(),
        torch.zeros_like(reference64),
    ).contiguous()
    coordinate = _first_argmax_coordinate(swallowed_magnitude)
    return PACREVCCompleteSwallowObservation(
        name=name,
        gate_eligible=False,
        eligible_element_count=int(eligible.sum().item()),
        swallowed_element_count=int(swallowed.sum().item()),
        maximum_swallowed_reference_magnitude=float(
            torch.max(swallowed_magnitude).item()
        ),
        maximum_swallowed_coordinate=coordinate,
    )


@dataclass(frozen=True)
class PACREVCReadoutLaneDiagnostics:
    name: str
    readout_hex: tuple[str, ...]
    readout_exact_zero: bool
    actual_energy_error: PACREVCErrorSummary
    flipped_energy_error: PACREVCErrorSummary
    interaction_error: PACREVCErrorSummary
    field_error: PACREVCErrorSummary
    actual_energy_swallow: PACREVCCompleteSwallowObservation
    flipped_energy_swallow: PACREVCCompleteSwallowObservation
    interaction_swallow: PACREVCCompleteSwallowObservation

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "readout_hex": list(self.readout_hex),
            "readout_exact_zero": self.readout_exact_zero,
            "actual_energy_error": (
                self.actual_energy_error.canonical_payload()
            ),
            "flipped_energy_error": (
                self.flipped_energy_error.canonical_payload()
            ),
            "interaction_error": self.interaction_error.canonical_payload(),
            "field_error": self.field_error.canonical_payload(),
            "actual_energy_swallow": (
                self.actual_energy_swallow.canonical_payload()
            ),
            "flipped_energy_swallow": (
                self.flipped_energy_swallow.canonical_payload()
            ),
            "interaction_swallow": (
                self.interaction_swallow.canonical_payload()
            ),
        }


@dataclass(frozen=True)
class PACREVCOracleIntegrity:
    passed: bool
    formula_fixed: bool
    reference_finite: bool
    zero_readout_lane_exact: bool
    fixed_readout_policy_valid: bool
    model_state_unchanged: bool
    gradient_buffers_unchanged: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "formula_fixed": self.formula_fixed,
            "reference_finite": self.reference_finite,
            "zero_readout_lane_exact": self.zero_readout_lane_exact,
            "fixed_readout_policy_valid": (
                self.fixed_readout_policy_valid
            ),
            "model_state_unchanged": self.model_state_unchanged,
            "gradient_buffers_unchanged": self.gradient_buffers_unchanged,
            "reference_policy": PACRE_VC_FP64_ORACLE_POLICY,
            "fixed_readout_policy": PACRE_VC_FIXED_READOUT_POLICY,
            "error_threshold_gate_eligible": False,
            "swallow_threshold_gate_eligible": False,
        }


@dataclass(frozen=True)
class PACREVCOracleNumericalDiagnostics:
    actual_common_error: PACREVCErrorSummary
    actual_specific_error: PACREVCErrorSummary
    flipped_common_error: PACREVCErrorSummary
    flipped_specific_error: PACREVCErrorSummary
    actual_hidden_error: PACREVCErrorSummary
    flipped_hidden_error: PACREVCErrorSummary
    actual_joint_swallow: PACREVCCompleteSwallowObservation
    flipped_joint_swallow: PACREVCCompleteSwallowObservation
    actual_hidden_swallow: PACREVCCompleteSwallowObservation
    flipped_hidden_swallow: PACREVCCompleteSwallowObservation
    zero_readout: PACREVCReadoutLaneDiagnostics
    fixed_readout: PACREVCReadoutLaneDiagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "gate_eligible": False,
            "actual_common_error": (
                self.actual_common_error.canonical_payload()
            ),
            "actual_specific_error": (
                self.actual_specific_error.canonical_payload()
            ),
            "flipped_common_error": (
                self.flipped_common_error.canonical_payload()
            ),
            "flipped_specific_error": (
                self.flipped_specific_error.canonical_payload()
            ),
            "actual_hidden_error": (
                self.actual_hidden_error.canonical_payload()
            ),
            "flipped_hidden_error": (
                self.flipped_hidden_error.canonical_payload()
            ),
            "actual_joint_swallow": (
                self.actual_joint_swallow.canonical_payload()
            ),
            "flipped_joint_swallow": (
                self.flipped_joint_swallow.canonical_payload()
            ),
            "actual_hidden_swallow": (
                self.actual_hidden_swallow.canonical_payload()
            ),
            "flipped_hidden_swallow": (
                self.flipped_hidden_swallow.canonical_payload()
            ),
            "zero_readout": self.zero_readout.canonical_payload(),
            "fixed_readout": self.fixed_readout.canonical_payload(),
        }


@dataclass(frozen=True)
class PACREVCFP64OracleReport:
    integrity: PACREVCOracleIntegrity
    numerical: PACREVCOracleNumericalDiagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "policy": PACRE_VC_FP64_ORACLE_POLICY,
            "integrity": self.integrity.canonical_payload(),
            "numerical_diagnostics": self.numerical.canonical_payload(),
        }


def _gradient_snapshot(
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
) -> tuple[Tensor | None, ...]:
    return tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    )


def _gradients_equal(
    before: tuple[Tensor | None, ...],
    after: tuple[Tensor | None, ...],
) -> bool:
    if len(before) != len(after):
        return False
    return all(
        (first is None and second is None)
        or (
            isinstance(first, Tensor)
            and isinstance(second, Tensor)
            and torch.equal(first, second)
        )
        for first, second in zip(before, after, strict=True)
    )


def _readout_lane(
    *,
    name: str,
    readout32: Tensor,
    actual_hidden32: Tensor,
    flipped_hidden32: Tensor,
    actual_hidden64: Tensor,
    flipped_hidden64: Tensor,
    field_amplitude: float,
    stride: int,
) -> PACREVCReadoutLaneDiagnostics:
    if (
        readout32.device != actual_hidden32.device
        or readout32.dtype != torch.float32
        or readout32.shape != (actual_hidden32.shape[2],)
        or not readout32.is_contiguous()
        or not bool(torch.isfinite(readout32).all())
    ):
        raise ValueError("diagnostic readout differs from hidden tensors")
    weight32 = readout32[None, None, :, None, None]
    actual_energy32 = (
        actual_hidden32 * weight32
    ).sum(dim=2).contiguous()
    flipped_energy32 = (
        flipped_hidden32 * weight32
    ).sum(dim=2).contiguous()
    interaction32 = (
        0.5 * (actual_energy32 - flipped_energy32)
    ).contiguous()
    native_field32 = (field_amplitude + interaction32).contiguous()
    field32 = F.pixel_shuffle(native_field32, stride).contiguous()

    readout64 = readout32.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    weight64 = readout64[None, None, :, None, None]
    actual_energy64 = (
        actual_hidden64 * weight64
    ).sum(dim=2).contiguous()
    flipped_energy64 = (
        flipped_hidden64 * weight64
    ).sum(dim=2).contiguous()
    interaction64 = (
        0.5 * (actual_energy64 - flipped_energy64)
    ).contiguous()
    amplitude32 = torch.tensor(
        field_amplitude,
        dtype=torch.float32,
    )
    amplitude64 = float(amplitude32.to(torch.float64).item())
    native_field64 = (amplitude64 + interaction64).contiguous()
    field64 = F.pixel_shuffle(native_field64, stride).contiguous()

    return PACREVCReadoutLaneDiagnostics(
        name=name,
        readout_hex=tuple(
            float(value).hex()
            for value in readout64.tolist()
        ),
        readout_exact_zero=bool(torch.count_nonzero(readout32) == 0),
        actual_energy_error=_error_summary(
            actual_energy32,
            actual_energy64,
        ),
        flipped_energy_error=_error_summary(
            flipped_energy32,
            flipped_energy64,
        ),
        interaction_error=_error_summary(
            interaction32,
            interaction64,
        ),
        field_error=_error_summary(field32, field64),
        actual_energy_swallow=complete_swallow_observation(
            f"{name}_actual_energy_complete_swallow",
            actual_energy32,
            actual_energy64,
        ),
        flipped_energy_swallow=complete_swallow_observation(
            f"{name}_flipped_energy_complete_swallow",
            flipped_energy32,
            flipped_energy64,
        ),
        interaction_swallow=complete_swallow_observation(
            f"{name}_interaction_complete_swallow",
            interaction32,
            interaction64,
        ),
    )


@torch.no_grad()
def run_pacre_fp64_oracle(
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    fields: CoverageStatePACREFields,
) -> PACREVCFP64OracleReport:
    """Run integrity checks and separate diagnostic-only FP64 comparisons."""

    validate_pacre_fields_contract(model, fields)
    state_before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    gradients_before = _gradient_snapshot(model)

    occupancy64 = fields.occupancy_affine.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).unsqueeze(1)
    flipped_occupancy64 = fields.flipped_occupancy_affine.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    phase64 = fields.phase_feature_affine.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    mean64 = fields.phase_feature_mean.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    actual_common64 = (
        occupancy64 + mean64
    ).expand_as(phase64).contiguous()
    actual_specific64 = (occupancy64 + phase64).contiguous()
    flipped_common64 = (
        flipped_occupancy64 + mean64
    ).contiguous()
    flipped_specific64 = (
        flipped_occupancy64 + phase64
    ).contiguous()
    actual_hidden64 = (
        F.silu(actual_specific64) - F.silu(actual_common64)
    ).contiguous()
    flipped_hidden64 = (
        F.silu(flipped_specific64) - F.silu(flipped_common64)
    ).contiguous()
    reference_tensors = (
        actual_common64,
        actual_specific64,
        flipped_common64,
        flipped_specific64,
        actual_hidden64,
        flipped_hidden64,
    )
    reference_finite = all(
        bool(torch.isfinite(value).all()) for value in reference_tensors
    )
    if not reference_finite:
        raise FloatingPointError("FP64 oracle reference must be finite")

    actual_joint_delta32 = (
        fields.actual_specific_joint_affine
        - fields.actual_common_joint_affine
    ).contiguous()
    flipped_joint_delta32 = (
        fields.flipped_specific_joint_affine
        - fields.flipped_common_joint_affine
    ).contiguous()
    actual_joint_delta64 = (
        actual_specific64 - actual_common64
    ).contiguous()
    flipped_joint_delta64 = (
        flipped_specific64 - flipped_common64
    ).contiguous()

    zero_readout32 = torch.zeros(
        int(model.config.width),
        dtype=torch.float32,
        device=fields.field.device,
    )
    fixed_readout32 = torch.linspace(
        0.5,
        1.5,
        int(model.config.width),
        dtype=torch.float32,
        device="cpu",
    ).to(device=fields.field.device)
    zero_lane = _readout_lane(
        name="zero_readout",
        readout32=zero_readout32,
        actual_hidden32=fields.actual_compatibility_hidden,
        flipped_hidden32=fields.flipped_compatibility_hidden,
        actual_hidden64=actual_hidden64,
        flipped_hidden64=flipped_hidden64,
        field_amplitude=float(model.config.field_amplitude),
        stride=int(model.config.feature_stride),
    )
    fixed_lane = _readout_lane(
        name="fixed_linspace_readout",
        readout32=fixed_readout32,
        actual_hidden32=fields.actual_compatibility_hidden,
        flipped_hidden32=fields.flipped_compatibility_hidden,
        actual_hidden64=actual_hidden64,
        flipped_hidden64=flipped_hidden64,
        field_amplitude=float(model.config.field_amplitude),
        stride=int(model.config.feature_stride),
    )

    actual_joint_prerequisite = actual_joint_delta64.ne(0.0)
    flipped_joint_prerequisite = flipped_joint_delta64.ne(0.0)
    numerical = PACREVCOracleNumericalDiagnostics(
        actual_common_error=_error_summary(
            fields.actual_common_joint_affine,
            actual_common64,
        ),
        actual_specific_error=_error_summary(
            fields.actual_specific_joint_affine,
            actual_specific64,
        ),
        flipped_common_error=_error_summary(
            fields.flipped_common_joint_affine,
            flipped_common64,
        ),
        flipped_specific_error=_error_summary(
            fields.flipped_specific_joint_affine,
            flipped_specific64,
        ),
        actual_hidden_error=_error_summary(
            fields.actual_compatibility_hidden,
            actual_hidden64,
        ),
        flipped_hidden_error=_error_summary(
            fields.flipped_compatibility_hidden,
            flipped_hidden64,
        ),
        actual_joint_swallow=complete_swallow_observation(
            "actual_joint_complete_swallow",
            actual_joint_delta32,
            actual_joint_delta64,
        ),
        flipped_joint_swallow=complete_swallow_observation(
            "flipped_joint_complete_swallow",
            flipped_joint_delta32,
            flipped_joint_delta64,
        ),
        actual_hidden_swallow=complete_swallow_observation(
            "actual_hidden_complete_swallow",
            fields.actual_compatibility_hidden,
            actual_hidden64,
            prerequisite=actual_joint_prerequisite,
        ),
        flipped_hidden_swallow=complete_swallow_observation(
            "flipped_hidden_complete_swallow",
            fields.flipped_compatibility_hidden,
            flipped_hidden64,
            prerequisite=flipped_joint_prerequisite,
        ),
        zero_readout=zero_lane,
        fixed_readout=fixed_lane,
    )

    state_after = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    gradients_after = _gradient_snapshot(model)
    state_unchanged = all(
        torch.equal(first, second)
        for first, second in zip(state_before, state_after, strict=True)
    )
    gradients_unchanged = _gradients_equal(
        gradients_before,
        gradients_after,
    )
    zero_exact = (
        zero_lane.readout_exact_zero
        and zero_lane.actual_energy_error.maximum_absolute_error == 0.0
        and zero_lane.flipped_energy_error.maximum_absolute_error == 0.0
        and zero_lane.interaction_error.maximum_absolute_error == 0.0
        and zero_lane.field_error.maximum_absolute_error == 0.0
    )
    fixed_policy_valid = (
        fixed_readout32.shape == (int(model.config.width),)
        and bool(torch.all(fixed_readout32 > 0.0))
        and float(fixed_readout32[0].detach().cpu()) == 0.5
        and float(fixed_readout32[-1].detach().cpu()) == 1.5
    )
    integrity = PACREVCOracleIntegrity(
        passed=(
            reference_finite
            and zero_exact
            and fixed_policy_valid
            and state_unchanged
            and gradients_unchanged
        ),
        formula_fixed=True,
        reference_finite=reference_finite,
        zero_readout_lane_exact=zero_exact,
        fixed_readout_policy_valid=fixed_policy_valid,
        model_state_unchanged=state_unchanged,
        gradient_buffers_unchanged=gradients_unchanged,
    )
    return PACREVCFP64OracleReport(
        integrity=integrity,
        numerical=numerical,
    )


__all__ = [
    "PACRE_VC_DIAGNOSTIC_POLICY",
    "PACRE_VC_FIXED_READOUT_POLICY",
    "PACRE_VC_FP64_ORACLE_POLICY",
    "PACRE_VC_LEGACY_ATOL",
    "PACRE_VC_LEGACY_RTOL",
    "PACRE_VC_NORMALIZED_ERROR_POLICY",
    "PACRE_VC_RELATIVE_ERROR_POLICY",
    "PACRE_VC_ULP_POLICY",
    "PACREVCCompleteSwallowObservation",
    "PACREVCErrorSummary",
    "PACREVCFP64OracleReport",
    "PACREVCLegacyDiagnostics",
    "PACREVCLegacyResidualLane",
    "PACREVCOracleIntegrity",
    "PACREVCOracleNumericalDiagnostics",
    "PACREVCReadoutLaneDiagnostics",
    "complete_swallow_observation",
    "fp32_ulp_distance",
    "legacy_residual_lane",
    "legacy_subtraction_diagnostics",
    "normalized_absolute_error",
    "reference_relative_error",
    "run_pacre_fp64_oracle",
]
