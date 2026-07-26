from __future__ import annotations

import pytest
import torch

from cure_lite.paired_losses import PairedDifferenceLoss


def _manual_loss(
    logits_plus: torch.Tensor,
    logits_minus: torch.Tensor,
    increment: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    delta = torch.sigmoid(logits_minus) - torch.sigmoid(logits_plus)
    positive = increment.to(torch.bool)
    zero = valid & ~positive
    states = []
    for index in range(delta.shape[0]):
        states.append(
            0.5 * (((delta[index][positive[index]] - 1.0) / 2.0) ** 2).mean()
            + 0.5 * (delta[index][zero[index]] ** 2).mean()
        )
    return torch.stack(states).mean()


def test_paired_difference_loss_matches_frozen_equation_per_pair() -> None:
    logits_plus = torch.tensor(
        [
            [[[0.1, -0.7, 1.2, -1.1]]],
            [[[0.4, 0.8, -0.2, -0.9]]],
        ],
        dtype=torch.float32,
    )
    logits_minus = torch.tensor(
        [
            [[[1.4, -0.1, 0.3, -0.5]]],
            [[[0.2, 1.7, 0.6, -1.3]]],
        ],
        dtype=torch.float32,
    )
    increment = torch.tensor(
        [
            [[[1.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    valid = torch.ones_like(increment, dtype=torch.bool)

    result = PairedDifferenceLoss()(
        logits_plus,
        logits_minus,
        increment,
        valid,
    )

    expected = _manual_loss(logits_plus, logits_minus, increment, valid)
    torch.testing.assert_close(result["total"], expected, rtol=0.0, atol=0.0)
    assert int(result["pair_count"]) == 2
    assert int(result["positive_response_pixels"]) == 2
    assert int(result["zero_response_pixels"]) == 6
    assert tuple(result["per_pair_total"].shape) == (2,)


def test_zero_response_pixels_are_class_balanced_not_pixel_count_weighted() -> None:
    logits_plus = torch.tensor([[[[0.2, -0.4]]]])
    logits_minus = torch.tensor([[[[0.8, 0.6]]]])
    increment = torch.tensor([[[[1.0, 0.0]]]])
    valid = torch.ones_like(increment, dtype=torch.bool)
    base = PairedDifferenceLoss()(
        logits_plus,
        logits_minus,
        increment,
        valid,
    )["total"]

    duplicated = PairedDifferenceLoss()(
        torch.cat((logits_plus[..., :1], logits_plus[..., 1:].repeat(1, 1, 1, 7)), dim=-1),
        torch.cat(
            (logits_minus[..., :1], logits_minus[..., 1:].repeat(1, 1, 1, 7)),
            dim=-1,
        ),
        torch.cat((increment[..., :1], increment[..., 1:].repeat(1, 1, 1, 7)), dim=-1),
        torch.ones(1, 1, 1, 8, dtype=torch.bool),
    )["total"]

    torch.testing.assert_close(duplicated, base, rtol=1e-7, atol=1e-8)


def test_both_endpoint_logits_keep_finite_nonzero_gradients() -> None:
    logits_plus = torch.tensor([[[[0.2, -0.4, 0.8]]]], requires_grad=True)
    logits_minus = torch.tensor([[[[1.0, 0.3, -0.1]]]], requires_grad=True)
    increment = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    valid = torch.ones_like(increment, dtype=torch.bool)

    PairedDifferenceLoss()(
        logits_plus,
        logits_minus,
        increment,
        valid,
    )["total"].backward()

    assert logits_plus.grad is not None
    assert logits_minus.grad is not None
    assert torch.isfinite(logits_plus.grad).all()
    assert torch.isfinite(logits_minus.grad).all()
    assert torch.any(logits_plus.grad != 0.0)
    assert torch.any(logits_minus.grad != 0.0)


def test_production_loss_has_nonzero_cross_endpoint_mixed_derivative() -> None:
    logits_plus = torch.tensor(
        [[[[0.2, -0.4, 0.8]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    logits_minus = torch.tensor(
        [[[[1.0, 0.3, -0.1]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    increment = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    valid = torch.ones_like(increment, dtype=torch.bool)

    loss = PairedDifferenceLoss()(
        logits_plus,
        logits_minus,
        increment,
        valid,
    )["total"]
    grad_minus = torch.autograd.grad(
        loss,
        logits_minus,
        create_graph=True,
    )[0]
    mixed = torch.autograd.grad(grad_minus.sum(), logits_plus)[0]

    assert torch.isfinite(mixed).all()
    assert torch.count_nonzero(mixed) == mixed.numel()


def test_equal_raw_pre_mask_logits_have_zero_delta_not_mask_induced_delta() -> None:
    logits = torch.full((1, 1, 2, 2), 0.7)
    increment = torch.zeros_like(logits)
    increment[0, 0, 0, 0] = 1.0
    valid = torch.ones_like(logits, dtype=torch.bool)

    result = PairedDifferenceLoss()(logits, logits, increment, valid)

    # delta is exactly zero everywhere: the positive term is
    # 0.5 * ((0 - 1) / 2)^2 and the zero-response term is zero.
    torch.testing.assert_close(
        result["total"],
        torch.tensor(0.125),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result["zero_stratum_mse"],
        torch.tensor(0.0),
        rtol=0.0,
        atol=0.0,
    )


def test_pair_identity_changes_coupled_loss_with_fixed_endpoint_marginals() -> None:
    logits_plus = torch.tensor(
        [
            [[[-1.7, 0.2, 0.5]]],
            [[[0.9, -0.8, 1.3]]],
        ]
    )
    logits_minus = torch.tensor(
        [
            [[[1.6, -0.5, 0.1]]],
            [[[-1.2, 1.5, 0.7]]],
        ]
    )
    increment = torch.tensor(
        [
            [[[1.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0]]],
        ]
    )
    valid = torch.ones_like(increment, dtype=torch.bool)
    criterion = PairedDifferenceLoss()

    aligned = criterion(logits_plus, logits_minus, increment, valid)["total"]
    permuted = criterion(logits_plus, logits_minus.flip(0), increment, valid)["total"]

    assert not torch.isclose(aligned, permuted)


@pytest.mark.parametrize("missing_stratum", ["positive", "zero"])
def test_every_pair_requires_both_response_strata(missing_stratum: str) -> None:
    logits = torch.zeros(1, 1, 2, 2)
    if missing_stratum == "positive":
        increment = torch.zeros_like(logits)
        valid = torch.ones_like(logits, dtype=torch.bool)
        message = "non-empty response"
    else:
        increment = torch.ones_like(logits)
        valid = torch.ones_like(logits, dtype=torch.bool)
        message = "zero-response"

    with pytest.raises(ValueError, match=message):
        PairedDifferenceLoss()(logits, logits, increment, valid)


def test_increment_outside_valid_domain_is_rejected() -> None:
    logits = torch.zeros(1, 1, 2, 2)
    increment = torch.zeros_like(logits)
    increment[0, 0, 0, 0] = 1.0
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[0, 0, 0, 0] = False

    with pytest.raises(ValueError, match="outside image_valid_mask"):
        PairedDifferenceLoss()(logits, logits, increment, valid)
