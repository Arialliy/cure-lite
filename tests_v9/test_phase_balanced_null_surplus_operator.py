from __future__ import annotations

import pytest
import torch

from cure_lite.phase_balanced_null_surplus_factorized_decoder import (
    phase_balanced_null_surplus_evidence,
)


def _operator(
    raw: torch.Tensor,
    count: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if count is None:
        count = torch.zeros(
            raw.shape[0],
            1,
            raw.shape[2],
            raw.shape[3],
            dtype=raw.dtype,
            device=raw.device,
        )
    return phase_balanced_null_surplus_evidence(raw, count)


def test_operator_matches_the_frozen_equation_exactly() -> None:
    raw = torch.tensor(
        [[[[1.2]], [[0.1]], [[-0.4]], [[-1.3]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    count = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    intensity, threshold, signed, active, evidence = _operator(
        raw,
        count,
    )

    expected_intensity = raw.exp()
    phase_count = raw.shape[1]
    expected_threshold = (
        phase_count + expected_intensity.sum(dim=1, keepdim=True)
    ) / (2 * phase_count)
    expected_signed = (
        expected_intensity - expected_threshold
    ) / (1.0 + count)
    expected_forward = expected_signed.clamp_min(0.0)

    assert torch.equal(intensity, expected_intensity)
    assert torch.equal(threshold, expected_threshold)
    assert torch.equal(signed, expected_signed)
    assert torch.equal(active, expected_signed > 0.0)
    assert torch.equal(evidence, expected_forward)

    upstream = torch.tensor(
        [[[[0.7]], [[-1.1]], [[1.9]], [[-0.3]]]],
        dtype=torch.float64,
    )
    (evidence * upstream).sum().backward()
    actual_gradient = raw.grad.detach().clone()

    comparison_raw = raw.detach().clone().requires_grad_(True)
    comparison_intensity = comparison_raw.exp()
    comparison_threshold = (
        phase_count
        + comparison_intensity.sum(dim=1, keepdim=True)
    ) / (2 * phase_count)
    comparison_signed = (
        comparison_intensity - comparison_threshold
    ) / (1.0 + count)
    (comparison_signed * upstream).sum().backward()
    torch.testing.assert_close(
        actual_gradient,
        comparison_raw.grad,
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_uniform_zero_is_the_exact_phase_balanced_null_anchor() -> None:
    raw = torch.zeros((2, 16, 3, 5), dtype=torch.float32)
    count = torch.tensor(
        [[[[0.0]]], [[[3.0]]]],
        dtype=torch.float32,
    ).expand(2, 1, 3, 5)
    intensity, threshold, signed, active, evidence = _operator(
        raw,
        count,
    )

    assert torch.equal(intensity, torch.ones_like(intensity))
    assert torch.equal(threshold, torch.ones_like(threshold))
    assert torch.equal(signed, torch.zeros_like(signed))
    assert not torch.any(active)
    assert torch.equal(evidence, torch.zeros_like(evidence))


def test_uniform_positive_can_activate_every_phase() -> None:
    raw = torch.full((1, 16, 2, 3), 0.4, dtype=torch.float64)
    count = torch.full((1, 1, 2, 3), 2.0, dtype=torch.float64)
    _, _, signed, active, evidence = _operator(raw, count)
    expected = (raw.exp() - 1.0) / (2.0 * (1.0 + count))

    assert torch.all(active)
    assert torch.all(evidence > 0.0)
    torch.testing.assert_close(evidence, expected)
    torch.testing.assert_close(signed, expected)


def test_uniform_negative_selects_null_for_every_phase() -> None:
    raw = torch.full((1, 16, 2, 3), -0.4, dtype=torch.float64)
    _, _, signed, active, evidence = _operator(raw)

    assert torch.all(signed < 0.0)
    assert not torch.any(active)
    assert torch.equal(evidence, torch.zeros_like(evidence))


def test_stride_one_case_is_nondegenerate() -> None:
    positive = torch.tensor([[[[0.7]]]], dtype=torch.float64)
    negative = torch.tensor([[[[-0.7]]]], dtype=torch.float64)

    positive_fields = _operator(positive)
    negative_fields = _operator(negative)
    expected_positive = (positive.exp() - 1.0) / 2.0

    torch.testing.assert_close(positive_fields[4], expected_positive)
    assert bool(positive_fields[3].item())
    assert float(negative_fields[2].item()) < 0.0
    assert not bool(negative_fields[3].item())
    assert float(negative_fields[4].item()) == 0.0


def test_inactive_phases_keep_a_finite_recovery_gradient() -> None:
    raw = torch.full(
        (1, 4, 1, 1),
        -2.0,
        dtype=torch.float64,
        requires_grad=True,
    )
    evidence = _operator(raw)[4]
    assert torch.equal(evidence, torch.zeros_like(evidence))

    evidence.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert torch.all(raw.grad > 0.0)
    torch.testing.assert_close(
        raw.grad,
        raw.detach().exp() / 2.0,
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_wrong_winner_objective_has_the_required_direction() -> None:
    raw = torch.tensor(
        [[[[2.0]], [[-1.0]], [[-2.0]], [[-3.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    evidence = _operator(raw)[4]
    wrong_phase = 0
    target_phase = 1
    assert float(evidence[0, wrong_phase].detach()) > 0.0
    assert float(evidence[0, target_phase].detach()) == 0.0

    loss = (
        evidence[0, wrong_phase, 0, 0]
        - evidence[0, target_phase, 0, 0]
    )
    loss.backward()

    assert float(raw.grad[0, wrong_phase]) > 0.0
    assert float(raw.grad[0, target_phase]) < 0.0
    torch.testing.assert_close(
        raw.grad[0, wrong_phase],
        raw.detach().exp()[0, wrong_phase],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        raw.grad[0, target_phase],
        -raw.detach().exp()[0, target_phase],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_common_direction_has_half_strength_at_the_null_anchor() -> None:
    common = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    raw = common.expand(1, 16, 1, 1)
    signed = _operator(raw)[2]

    signed[0, 0, 0, 0].backward()
    torch.testing.assert_close(
        common.grad,
        torch.tensor(0.5, dtype=torch.float64),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_occupancy_deletion_is_monotone_and_selection_invariant() -> None:
    raw = torch.tensor(
        [[[[1.4]], [[0.3]], [[-0.2]], [[-1.1]]]],
        dtype=torch.float64,
    )
    occupied_count = torch.full(
        (1, 1, 1, 1),
        4.0,
        dtype=torch.float64,
    )
    deleted_count = torch.zeros_like(occupied_count)
    occupied = _operator(raw, occupied_count)
    deleted = _operator(raw, deleted_count)

    assert torch.equal(occupied[3], deleted[3])
    assert torch.all(deleted[4] >= occupied[4])
    assert torch.any(deleted[4] > occupied[4])


def test_positive_surplus_obeys_the_frozen_capacity_bound() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9031)
        raw = torch.randn(3, 16, 7, 5, dtype=torch.float64) * 2.5
        count = torch.randint(
            0,
            10,
            (3, 1, 7, 5),
        ).to(torch.float64)
    intensity, _, _, _, evidence = _operator(raw, count)
    phase_count = raw.shape[1]
    capacity = (
        phase_count + intensity.sum(dim=1, keepdim=True)
    ) / (1.0 + count)
    upper = (1.0 - 1.0 / (2.0 * phase_count)) * capacity

    assert torch.all(evidence >= 0.0)
    assert torch.all(evidence.sum(dim=1, keepdim=True) <= upper)


@pytest.mark.parametrize(
    ("raw", "count", "error"),
    [
        (
            torch.ones(1, 4, 1, 1, dtype=torch.int64),
            torch.zeros(1, 1, 1, 1),
            TypeError,
        ),
        (
            torch.full((1, 4, 1, 1), float("nan")),
            torch.zeros(1, 1, 1, 1),
            ValueError,
        ),
        (
            torch.zeros(1, 4, 1, 1),
            torch.full((1, 1, 1, 1), -1.0),
            ValueError,
        ),
        (
            torch.zeros(1, 4, 1, 1),
            torch.zeros(1, 1, 2, 1),
            ValueError,
        ),
    ],
)
def test_operator_rejects_invalid_fields(
    raw: torch.Tensor,
    count: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        phase_balanced_null_surplus_evidence(raw, count)
