from __future__ import annotations

import inspect

import pytest
import torch

from cure_lite.config import LossConfig
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_outcome_losses import OutcomeCompleteTransitionLoss


def _mixed_inputs() -> dict[str, torch.Tensor]:
    shape = (2, 1, 4, 5)
    count = int(torch.tensor(shape).prod())
    logits_plus = torch.linspace(-1.1, 0.9, steps=count).reshape(shape)
    logits_minus = torch.linspace(0.8, -0.7, steps=count).reshape(shape)
    logits_plus.requires_grad_()
    logits_minus.requires_grad_()

    valid = torch.ones(shape, dtype=torch.bool)
    occupancy = torch.zeros(shape, dtype=torch.bool)
    gt_union = torch.zeros(shape, dtype=torch.bool)
    completion = torch.zeros(shape, dtype=torch.bool)
    increment = torch.zeros(shape, dtype=torch.float32)
    footprint = torch.zeros(shape, dtype=torch.bool)

    # Pair 0 has all D/H/G strata.
    increment[0, 0, 0, 0] = 1.0
    footprint[0, 0, 0, 0:2] = True
    completion[0, 0, 1, 1] = True
    occupancy[0, 0, 3, 4] = True
    gt_union[0] |= completion[0]
    gt_union[0] |= increment[0].to(dtype=torch.bool)

    # Pair 1 is an empty-D outcome with both H and G active.
    footprint[1, 0, 2, 2:4] = True
    completion[1, 0, 1, 2] = True
    occupancy[1, 0, 3, 0] = True
    gt_union[1] |= completion[1]
    return {
        "logits_plus": logits_plus,
        "logits_minus": logits_minus,
        "completion_plus": completion,
        "occupancy_plus": occupancy,
        "gt_union": gt_union,
        "label_increment": increment,
        "image_valid_mask": valid,
        "intervention_footprint": footprint,
    }


def _slice(
    values: dict[str, torch.Tensor],
    index: int,
) -> dict[str, torch.Tensor]:
    return {name: value[index : index + 1] for name, value in values.items()}


def _manual_stratum_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return values[mask].mean()


def test_clean_hierarchical_formula_and_diagnostics_are_exact() -> None:
    config = LossConfig(dice_weight=0.7, epsilon=1.0e-5)
    values = _slice(_mixed_inputs(), 0)
    result = OutcomeCompleteTransitionLoss(config)(**values)

    score_plus = torch.sigmoid(values["logits_plus"])
    score_minus = torch.sigmoid(values["logits_minus"])
    delta = score_minus - score_plus
    response = values["label_increment"].to(dtype=torch.bool)
    local = values["intervention_footprint"] & ~response
    global_context = (
        values["image_valid_mask"]
        & ~response
        & ~values["intervention_footprint"]
    )
    response_risk = _manual_stratum_mean(
        ((delta - 1.0) / 2.0).square(),
        response,
    )
    local_risk = _manual_stratum_mean(delta.square(), local)
    global_risk = _manual_stratum_mean(delta.square(), global_context)
    zero_risk = 0.5 * (local_risk + global_risk)
    transition = 0.5 * (response_risk + zero_risk)

    background = (
        values["image_valid_mask"]
        & ~values["occupancy_plus"]
        & ~values["gt_union"]
    )
    anchor_valid = values["completion_plus"] | background
    anchor = CURELiteLoss(config)(
        values["logits_plus"],
        values["completion_plus"].to(dtype=torch.float32),
        anchor_valid,
    )
    expected_total = 0.5 * anchor["total"] + 0.5 * transition

    torch.testing.assert_close(result["per_pair_response_stratum"][0], response_risk)
    torch.testing.assert_close(result["per_pair_local_zero_stratum"][0], local_risk)
    torch.testing.assert_close(result["per_pair_global_zero_stratum"][0], global_risk)
    torch.testing.assert_close(result["per_pair_zero_risk"][0], zero_risk)
    torch.testing.assert_close(result["per_pair_transition"][0], transition)
    torch.testing.assert_close(result["per_pair_plus_anchor"][0], anchor["total"])
    torch.testing.assert_close(result["total"], expected_total)
    torch.testing.assert_close(result["loss"], expected_total)
    assert result["response_pixels_per_pair"].tolist() == [1]
    assert result["local_zero_pixels_per_pair"].tolist() == [1]
    assert result["global_zero_pixels_per_pair"].tolist() == [18]
    assert result["response_active_per_pair"].tolist() == [True]
    assert result["zero_active_strata_per_pair"].tolist() == [2]
    assert result["transition_active_groups_per_pair"].tolist() == [2]
    assert torch.equal(result["response_stratum"], response)
    assert torch.equal(result["local_zero_stratum"], local)
    assert torch.equal(result["global_zero_stratum"], global_context)
    assert torch.equal(result["plus_anchor_valid_mask"], anchor_valid)


def test_component_empty_d_uses_only_hierarchical_zero_risk_without_nan() -> None:
    values = _slice(_mixed_inputs(), 1)
    result = OutcomeCompleteTransitionLoss(LossConfig())(**values)
    delta = torch.sigmoid(values["logits_minus"]) - torch.sigmoid(
        values["logits_plus"]
    )
    local = values["intervention_footprint"]
    global_context = (
        values["image_valid_mask"] & ~values["intervention_footprint"]
    )
    expected_local = delta.square()[local].mean()
    expected_global = delta.square()[global_context].mean()
    expected_zero = 0.5 * (expected_local + expected_global)

    assert torch.isfinite(result["total"])
    assert torch.isfinite(result["transition_loss"])
    assert result["response_pixels_per_pair"].tolist() == [0]
    assert result["response_active_per_pair"].tolist() == [False]
    assert result["transition_active_groups_per_pair"].tolist() == [1]
    assert float(result["per_pair_response_stratum"][0].detach()) == 0.0
    torch.testing.assert_close(result["per_pair_zero_risk"][0], expected_zero)
    torch.testing.assert_close(result["per_pair_transition"][0], expected_zero)


@pytest.mark.parametrize("empty_group", ("local", "global"))
def test_one_empty_zero_stratum_safely_uses_the_other(
    empty_group: str,
) -> None:
    shape = (1, 1, 2, 2)
    logits_plus = torch.tensor(
        [[[[0.1, -0.2], [0.3, -0.4]]]],
        requires_grad=True,
    )
    logits_minus = torch.tensor(
        [[[[0.7, 0.5], [-0.1, 0.2]]]],
        requires_grad=True,
    )
    valid = torch.ones(shape, dtype=torch.bool)
    response = torch.zeros(shape, dtype=torch.float32)
    response[0, 0, 0, 0] = 1.0
    footprint = torch.ones(shape, dtype=torch.bool)
    if empty_group == "local":
        footprint.zero_()
        footprint[0, 0, 0, 0] = True
    empty = torch.zeros(shape, dtype=torch.bool)
    gt_union = response.to(dtype=torch.bool)
    result = OutcomeCompleteTransitionLoss(LossConfig())(
        logits_plus,
        logits_minus,
        empty,
        empty,
        gt_union,
        response,
        valid,
        footprint,
    )
    delta_squared = (
        torch.sigmoid(logits_minus) - torch.sigmoid(logits_plus)
    ).square()
    response_mask = response.to(dtype=torch.bool)
    local = footprint & ~response_mask
    global_context = valid & ~response_mask & ~footprint
    active_zero = global_context if empty_group == "local" else local
    expected_zero = delta_squared[active_zero].mean()

    assert torch.isfinite(result["total"])
    assert result["zero_active_strata_per_pair"].tolist() == [1]
    if empty_group == "local":
        assert result["local_zero_active_per_pair"].tolist() == [False]
        assert result["global_zero_active_per_pair"].tolist() == [True]
    else:
        assert result["local_zero_active_per_pair"].tolist() == [True]
        assert result["global_zero_active_per_pair"].tolist() == [False]
    torch.testing.assert_close(result["per_pair_zero_risk"][0], expected_zero)


def test_local_zero_error_is_not_diluted_by_global_pixel_count() -> None:
    shape = (1, 1, 8, 8)
    logits_plus = torch.zeros(shape, requires_grad=True)
    logits_minus_data = torch.zeros(shape)
    footprint = torch.zeros(shape, dtype=torch.bool)
    footprint[0, 0, 0, 0] = True
    logits_minus_data[footprint] = 10.0
    logits_minus = logits_minus_data.requires_grad_()
    valid = torch.ones(shape, dtype=torch.bool)
    empty_bool = torch.zeros(shape, dtype=torch.bool)
    empty_target = torch.zeros(shape, dtype=torch.float32)
    result = OutcomeCompleteTransitionLoss(LossConfig())(
        logits_plus,
        logits_minus,
        empty_bool,
        empty_bool,
        empty_bool,
        empty_target,
        valid,
        footprint,
    )
    delta = torch.sigmoid(logits_minus) - torch.sigmoid(logits_plus)
    local_risk = delta.square()[footprint].mean()
    global_risk = delta.square()[valid & ~footprint].mean()
    pooled_global_mean = delta.square()[valid].mean()

    torch.testing.assert_close(
        result["transition_loss"],
        0.5 * (local_risk + global_risk),
    )
    assert float(result["transition_loss"].detach()) > 0.12
    assert float(pooled_global_mean.detach()) < 0.004


def test_mixed_batch_is_the_ordinary_mean_of_pair_risks() -> None:
    values = _mixed_inputs()
    criterion = OutcomeCompleteTransitionLoss(LossConfig())
    mixed = criterion(**values)
    clean = criterion(**_slice(values, 0))
    component = criterion(**_slice(values, 1))

    expected = 0.5 * (clean["total"] + component["total"])
    torch.testing.assert_close(mixed["total"], expected)
    torch.testing.assert_close(mixed["per_pair_total"][0], clean["total"])
    torch.testing.assert_close(mixed["per_pair_total"][1], component["total"])
    torch.testing.assert_close(
        mixed["per_pair_transition"][0],
        clean["transition_loss"],
    )
    torch.testing.assert_close(
        mixed["per_pair_transition"][1],
        component["transition_loss"],
    )


def test_endpoint_gradients_and_mixed_derivative_remain_attached() -> None:
    values = _slice(_mixed_inputs(), 0)
    result = OutcomeCompleteTransitionLoss(LossConfig())(**values)
    plus_gradient = torch.autograd.grad(
        result["total"],
        values["logits_plus"],
        create_graph=True,
        retain_graph=True,
    )[0]
    minus_gradient = torch.autograd.grad(
        result["total"],
        values["logits_minus"],
        retain_graph=True,
    )[0]
    assert torch.isfinite(plus_gradient).all()
    assert torch.isfinite(minus_gradient).all()
    assert torch.count_nonzero(plus_gradient) > 0
    assert torch.count_nonzero(minus_gradient) > 0

    index = tuple(
        torch.nonzero(
            values["label_increment"],
            as_tuple=False,
        )[0].tolist()
    )
    mixed = torch.autograd.grad(
        plus_gradient[index],
        values["logits_minus"],
    )[0]
    assert torch.isfinite(mixed).all()
    assert float(mixed[index].abs()) > 0.0


def test_pair_permutation_preserves_total_and_permutes_diagnostics() -> None:
    values = _mixed_inputs()
    criterion = OutcomeCompleteTransitionLoss(LossConfig())
    original = criterion(**values)
    permutation = torch.tensor([1, 0])
    permuted_values = {
        name: value[permutation]
        for name, value in values.items()
    }
    permuted = criterion(**permuted_values)

    torch.testing.assert_close(permuted["total"], original["total"])
    for name in (
        "per_pair_total",
        "per_pair_plus_anchor",
        "per_pair_transition",
        "per_pair_zero_risk",
        "per_pair_response_stratum",
        "per_pair_local_zero_stratum",
        "per_pair_global_zero_stratum",
        "response_pixels_per_pair",
        "local_zero_pixels_per_pair",
        "global_zero_pixels_per_pair",
        "zero_active_strata_per_pair",
        "transition_active_groups_per_pair",
    ):
        torch.testing.assert_close(
            permuted[name],
            original[name][permutation],
        )


def test_loss_api_has_no_pair_kind_or_tunable_weight() -> None:
    init_parameters = set(
        inspect.signature(OutcomeCompleteTransitionLoss).parameters
    )
    forward_parameters = set(
        inspect.signature(OutcomeCompleteTransitionLoss.forward).parameters
    )
    assert init_parameters == {"config"}
    assert "pair_kind" not in forward_parameters
    assert not any("weight" in name for name in forward_parameters)
    with pytest.raises(TypeError, match="LossConfig"):
        OutcomeCompleteTransitionLoss(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutator", "error", "message"),
    (
        (
            lambda values: values.update(
                logits_minus=values["logits_minus"][:, :, :, :-1]
            ),
            ValueError,
            "identical shapes",
        ),
        (
            lambda values: values.update(
                label_increment=values["label_increment"].to(torch.float64)
            ),
            TypeError,
            "label_increment must be float32",
        ),
        (
            lambda values: values.update(
                intervention_footprint=values[
                    "intervention_footprint"
                ].to(torch.float32)
            ),
            TypeError,
            "intervention_footprint must be bool",
        ),
        (
            lambda values: values["intervention_footprint"].zero_(),
            ValueError,
            "non-empty intervention footprint",
        ),
        (
            lambda values: values["image_valid_mask"].__setitem__(
                (0, 0, 0, 1),
                False,
            ),
            ValueError,
            "intervention_footprint lies outside",
        ),
        (
            lambda values: values["gt_union"].__setitem__(
                (0, 0, 0, 0),
                False,
            ),
            ValueError,
            "label_increment must contain only GT",
        ),
    ),
)
def test_shape_dtype_and_semantic_errors_fail_closed(
    mutator,
    error,
    message: str,
) -> None:
    values = _mixed_inputs()
    mutator(values)
    with pytest.raises(error, match=message):
        OutcomeCompleteTransitionLoss(LossConfig())(**values)


def test_every_pair_requires_a_nonempty_zero_response_group() -> None:
    shape = (1, 1, 2, 2)
    logits_plus = torch.zeros(shape)
    logits_minus = torch.ones(shape)
    valid = torch.ones(shape, dtype=torch.bool)
    response = torch.ones(shape, dtype=torch.float32)
    footprint = torch.ones(shape, dtype=torch.bool)
    empty = torch.zeros(shape, dtype=torch.bool)
    with pytest.raises(ValueError, match="zero-response stratum"):
        OutcomeCompleteTransitionLoss(LossConfig())(
            logits_plus,
            logits_minus,
            empty,
            empty,
            valid,
            response,
            valid,
            footprint,
        )
