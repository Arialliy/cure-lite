from __future__ import annotations

import pytest
import torch

from cure_lite_v23.algebra_verifier import (
    EPS32,
    EPS64,
    TINY32,
    gamma,
    phase_roundoff_observations,
)


def _centered_phase(
    *,
    phases: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(230201)
    phase = torch.randn(
        (2, phases, 3, 2, 3),
        generator=generator,
        dtype=torch.float32,
    ).contiguous()
    mean = phase.mean(dim=1, keepdim=True).contiguous()
    residual = (phase - mean).contiguous()
    return phase, mean, residual


def test_frozen_constants_and_gamma_are_receipt_stable() -> None:
    assert EPS32 == float(torch.finfo(torch.float32).eps)
    assert EPS64 == float(torch.finfo(torch.float64).eps)
    assert TINY32 == float(torch.finfo(torch.float32).tiny)
    assert TINY32 > float(
        torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(float("inf"), dtype=torch.float32),
        )
    )
    assert gamma(16, EPS32) == (
        16.0 * EPS32 / (1.0 - 16.0 * EPS32)
    )
    with pytest.raises(ValueError, match="invalid"):
        gamma(0, EPS32)
    with pytest.raises(ValueError, match="invalid"):
        gamma(True, EPS32)


def test_generated_phase_satisfies_both_ftz_safe_bounds() -> None:
    phase, mean, residual = _centered_phase()
    phase_before = phase.clone()
    mean_before = mean.clone()
    residual_before = residual.clone()

    report = phase_roundoff_observations(phase, mean, residual)

    assert report.passed
    assert report.reconstruction.failed_element_count == 0
    assert report.centering.failed_element_count == 0
    assert report.reconstruction.maximum_bound > 0.0
    assert report.centering.maximum_bound > 0.0
    assert torch.equal(phase, phase_before)
    assert torch.equal(mean, mean_before)
    assert torch.equal(residual, residual_before)
    payload = report.canonical_payload()
    assert payload["ftz_safe_floor"] is True
    assert payload["tiny32_hex"] == TINY32.hex()


def test_reconstruction_bound_matches_the_unique_frozen_formula() -> None:
    phase, mean, residual = _centered_phase(phases=4)
    report = phase_roundoff_observations(phase, mean, residual)

    x64 = phase.to(torch.float64)
    m64 = mean.to(torch.float64).expand_as(x64)
    expected_bound = (
        EPS32 * (x64.abs() + m64.abs()) + 2.0 * TINY32
    )

    assert report.reconstruction.maximum_bound == float(
        expected_bound.max()
    )
    assert report.reconstruction.passed


def test_tiny32_floor_covers_a_simulated_ftz_subtraction() -> None:
    values = torch.tensor(
        [
            0.5 * TINY32,
            -0.5 * TINY32,
            0.25 * TINY32,
            -0.25 * TINY32,
        ],
        dtype=torch.float32,
    ).reshape(1, 4, 1, 1, 1)
    phase = values.contiguous()
    mean = torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32)
    # This deliberately models an execution lane that flushed every
    # subnormal residual to zero.
    flushed_residual = torch.zeros_like(phase)

    report = phase_roundoff_observations(
        phase,
        mean,
        flushed_residual,
    )

    assert report.reconstruction.passed
    assert report.centering.passed
    assert report.reconstruction.maximum_error <= 0.5 * TINY32
    assert report.reconstruction.bound_at_maximum_error >= 2.0 * TINY32


def test_material_residual_corruption_fails_reconstruction_and_centering() -> None:
    phase, mean, residual = _centered_phase()
    corrupted = residual.clone()
    coordinate = (0, 3, 1, 1, 2)
    corrupted[coordinate] += 1.0e-3

    report = phase_roundoff_observations(phase, mean, corrupted)

    assert not report.passed
    assert not report.reconstruction.passed
    assert not report.centering.passed
    assert report.reconstruction.failed_element_count >= 1
    assert report.centering.failed_element_count >= 1
    assert report.reconstruction.argmax_coordinate == coordinate


def test_phase_bound_contract_rejects_dtype_shape_and_layout_drift() -> None:
    phase, mean, residual = _centered_phase()
    noncontiguous = (
        residual.transpose(-1, -2)
        .contiguous()
        .transpose(-1, -2)
    )
    assert not noncontiguous.is_contiguous()

    with pytest.raises(TypeError, match="FP32"):
        phase_roundoff_observations(
            phase.to(torch.float64),
            mean,
            residual,
        )
    with pytest.raises(ValueError, match="shapes"):
        phase_roundoff_observations(
            phase,
            mean,
            residual[..., :-1].contiguous(),
        )
    with pytest.raises(ValueError, match="contiguous"):
        phase_roundoff_observations(
            phase,
            mean,
            noncontiguous,
        )
