from __future__ import annotations

import pytest
import torch

from cure_lite.losses import CURELiteLoss
from cure_lite.paired_losses import PairedDifferenceLoss
from cure_lite.paired_transition_losses import AnchoredTransitionLoss


def _valid_inputs(
    *,
    batch_size: int = 2,
) -> dict[str, torch.Tensor]:
    shape = (batch_size, 1, 4, 5)
    logits_plus = torch.linspace(-1.2, 1.1, steps=torch.tensor(shape).prod().item())
    logits_plus = logits_plus.reshape(shape).requires_grad_()
    logits_minus = torch.linspace(0.9, -0.8, steps=torch.tensor(shape).prod().item())
    logits_minus = logits_minus.reshape(shape).requires_grad_()
    valid = torch.ones(shape, dtype=torch.bool)
    occupancy = torch.zeros(shape, dtype=torch.bool)
    gt_union = torch.zeros(shape, dtype=torch.bool)
    completion = torch.zeros(shape, dtype=torch.bool)
    increment = torch.zeros(shape, dtype=torch.float32)
    for index in range(batch_size):
        completion[index, 0, 0, index] = True
        increment[index, 0, 2, index + 1] = 1.0
    gt_union |= completion
    gt_union |= increment.to(dtype=torch.bool)
    return {
        "logits_plus": logits_plus,
        "logits_minus": logits_minus,
        "completion_plus": completion,
        "occupancy_plus": occupancy,
        "gt_union": gt_union,
        "label_increment": increment,
        "image_valid_mask": valid,
    }


def test_formula_matches_existing_component_losses() -> None:
    values = _valid_inputs()
    result = AnchoredTransitionLoss()(**values)

    background = (
        values["image_valid_mask"]
        & ~values["occupancy_plus"]
        & ~values["gt_union"]
    )
    anchor_valid = values["completion_plus"] | background
    anchor = CURELiteLoss()(
        values["logits_plus"],
        values["completion_plus"].to(dtype=torch.float32),
        anchor_valid,
    )
    transition = PairedDifferenceLoss()(
        values["logits_plus"],
        values["logits_minus"],
        values["label_increment"],
        values["image_valid_mask"],
    )
    expected_per_pair = (
        0.5 * anchor["per_state_total"]
        + 0.5 * transition["per_pair_total"]
    )

    torch.testing.assert_close(result["per_pair_plus_anchor"], anchor["per_state_total"])
    torch.testing.assert_close(
        result["per_pair_transition"],
        transition["per_pair_total"],
    )
    torch.testing.assert_close(result["per_pair_total"], expected_per_pair)
    torch.testing.assert_close(result["total"], expected_per_pair.mean())
    torch.testing.assert_close(result["loss"], result["total"])
    torch.testing.assert_close(
        result["plus_anchor_loss"],
        anchor["per_state_total"].mean(),
    )
    torch.testing.assert_close(
        result["transition_loss"],
        transition["per_pair_total"].mean(),
    )
    assert torch.equal(result["plus_anchor_background"], background)
    assert torch.equal(result["plus_anchor_valid_mask"], anchor_valid)
    assert int(result["pair_count"]) == 2


def test_both_endpoints_receive_nonzero_gradient() -> None:
    values = _valid_inputs()
    result = AnchoredTransitionLoss()(**values)
    result["total"].backward()

    plus_grad = values["logits_plus"].grad
    minus_grad = values["logits_minus"].grad
    assert plus_grad is not None
    assert minus_grad is not None
    assert torch.isfinite(plus_grad).all()
    assert torch.isfinite(minus_grad).all()
    assert torch.count_nonzero(plus_grad) > 0
    assert torch.count_nonzero(minus_grad) > 0


def test_objective_has_nonzero_mixed_endpoint_derivative() -> None:
    values = _valid_inputs(batch_size=1)
    result = AnchoredTransitionLoss()(**values)
    plus_gradient = torch.autograd.grad(
        result["total"],
        values["logits_plus"],
        create_graph=True,
    )[0]
    index = tuple(
        torch.nonzero(values["label_increment"], as_tuple=False)[0].tolist()
    )
    mixed = torch.autograd.grad(
        plus_gradient[index],
        values["logits_minus"],
    )[0]

    assert torch.isfinite(mixed).all()
    assert float(mixed[index].abs()) > 0.0


def test_empty_completion_plus_is_a_negative_only_anchor() -> None:
    values = _valid_inputs(batch_size=1)
    values["completion_plus"].zero_()
    values["gt_union"] = values["label_increment"].to(dtype=torch.bool)
    result = AnchoredTransitionLoss()(**values)
    expected_background = (
        values["image_valid_mask"]
        & ~values["occupancy_plus"]
        & ~values["gt_union"]
    )
    expected = CURELiteLoss()(
        values["logits_plus"],
        torch.zeros_like(values["label_increment"]),
        expected_background,
    )

    assert torch.isfinite(result["total"])
    assert torch.equal(result["plus_anchor_valid_mask"], expected_background)
    torch.testing.assert_close(
        result["plus_anchor_loss"],
        expected["total"],
    )


def test_unrelated_gt_is_excluded_from_plus_anchor_background() -> None:
    values = _valid_inputs(batch_size=1)
    third_gt = (0, 0, 3, 4)
    assert not bool(values["completion_plus"][third_gt])
    assert not bool(values["label_increment"][third_gt])
    values["gt_union"][third_gt] = True
    result = AnchoredTransitionLoss()(**values)

    assert not bool(result["plus_anchor_background"][third_gt])
    assert not bool(result["plus_anchor_valid_mask"][third_gt])


def test_batch_reduction_weights_pairs_equally() -> None:
    values = _valid_inputs()
    values["completion_plus"][1].zero_()
    values["gt_union"][1].zero_()
    values["label_increment"][1].zero_()
    values["completion_plus"][1, 0, 0, 0] = True
    values["label_increment"][1, 0, 1:4, 1:5] = 1.0
    values["gt_union"][1] |= values["completion_plus"][1]
    values["gt_union"][1] |= values["label_increment"][1].to(dtype=torch.bool)

    result = AnchoredTransitionLoss()(**values)
    first = AnchoredTransitionLoss()(
        **{name: tensor[:1] for name, tensor in values.items()}
    )
    second = AnchoredTransitionLoss()(
        **{name: tensor[1:] for name, tensor in values.items()}
    )
    expected = 0.5 * (first["total"] + second["total"])

    torch.testing.assert_close(result["total"], expected)
    torch.testing.assert_close(result["per_pair_total"][0], first["total"])
    torch.testing.assert_close(result["per_pair_total"][1], second["total"])


@pytest.mark.parametrize(
    ("mutator", "error", "message"),
    [
        (
            lambda values: values.update(
                completion_plus=values["completion_plus"].to(torch.float32)
            ),
            TypeError,
            "completion_plus must be bool",
        ),
        (
            lambda values: values["logits_plus"].__setitem__(
                (0, 0, 0, 0),
                float("nan"),
            ),
            ValueError,
            "logits_plus must be finite",
        ),
        (
            lambda values: values["image_valid_mask"].__setitem__(
                (0, 0, 3, 4),
                False,
            )
            or values["occupancy_plus"].__setitem__((0, 0, 3, 4), True),
            ValueError,
            "occupancy_plus lies outside",
        ),
        (
            lambda values: values["occupancy_plus"].__setitem__(
                tuple(torch.nonzero(values["completion_plus"])[0].tolist()),
                True,
            ),
            ValueError,
            "completion_plus must be valid and writable",
        ),
        (
            lambda values: values["gt_union"].__setitem__(
                tuple(torch.nonzero(values["completion_plus"])[0].tolist()),
                False,
            ),
            ValueError,
            "completion_plus must contain only GT pixels",
        ),
        (
            lambda values: values["gt_union"].__setitem__(
                tuple(torch.nonzero(values["label_increment"])[0].tolist()),
                False,
            ),
            ValueError,
            "label_increment must contain only GT pixels",
        ),
        (
            lambda values: values["label_increment"].__setitem__(
                tuple(torch.nonzero(values["completion_plus"])[0].tolist()),
                1.0,
            ),
            ValueError,
            "label_increment must be disjoint",
        ),
        (
            lambda values: values["label_increment"].zero_(),
            ValueError,
            "non-empty response stratum",
        ),
    ],
)
def test_invalid_inputs_fail_closed(mutator, error, message: str) -> None:
    values = _valid_inputs()
    if "logits_plus must be finite" in message:
        with torch.no_grad():
            mutator(values)
    else:
        mutator(values)
    with pytest.raises(error, match=message):
        AnchoredTransitionLoss()(**values)


def test_shape_dtype_and_device_contracts() -> None:
    values = _valid_inputs()
    values["logits_minus"] = values["logits_minus"][:, :, :, :-1]
    with pytest.raises(ValueError, match="identical shapes"):
        AnchoredTransitionLoss()(**values)

    values = _valid_inputs()
    values["label_increment"] = values["label_increment"].to(torch.float64)
    with pytest.raises(TypeError, match="label_increment must be float32"):
        AnchoredTransitionLoss()(**values)

    if torch.cuda.is_available():
        values = _valid_inputs()
        values["logits_minus"] = values["logits_minus"].cuda()
        with pytest.raises(ValueError, match="share a device"):
            AnchoredTransitionLoss()(**values)
