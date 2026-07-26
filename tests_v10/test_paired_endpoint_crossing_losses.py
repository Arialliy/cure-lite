from __future__ import annotations

import inspect

import pytest
import torch
from torch.nn import functional as F

from cure_lite.config import LossConfig
from cure_lite.conservative_factorized_decoder import (
    CURELiteConservativeFactorizedDecoder,
)
from cure_lite.experiment.conservative_toy_inputs import (
    SUPPORT_FAMILY,
    build_conservative_toy_case,
)
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_endpoint_crossing_losses import (
    PairedEndpointCrossingLoss,
)
from cure_lite.paired_outcome_losses import (
    OutcomeCompleteTransitionLoss,
)
from cure_lite.train.paired_outcome_step import (
    outcome_complete_train_step,
)


def _loss_inputs() -> dict[str, torch.Tensor]:
    shape = (2, 1, 3, 4)
    count = 2 * 3 * 4
    logits_plus = torch.linspace(-1.3, 1.1, count).reshape(shape)
    logits_minus = torch.linspace(0.9, -0.8, count).reshape(shape)
    logits_plus.requires_grad_()
    logits_minus.requires_grad_()

    valid = torch.ones(shape, dtype=torch.bool)
    occupancy = torch.zeros(shape, dtype=torch.bool)
    gt_union = torch.zeros(shape, dtype=torch.bool)
    completion = torch.zeros(shape, dtype=torch.bool)
    increment = torch.zeros(shape, dtype=torch.float32)
    footprint = torch.zeros(shape, dtype=torch.bool)

    # Pair 0 contains all D/H/G strata.
    increment[0, 0, 0, 0] = 1.0
    footprint[0, 0, 0, 0:2] = True
    completion[0, 0, 1, 1] = True
    occupancy[0, 0, 2, 3] = True
    gt_union[0] |= completion[0]
    gt_union[0] |= increment[0].to(dtype=torch.bool)

    # Pair 1 is an empty-D component-null-shaped loss input.
    footprint[1, 0, 0, 2:4] = True
    completion[1, 0, 1, 2] = True
    occupancy[1, 0, 2, 0] = True
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
    return {
        name: value[index : index + 1]
        for name, value in values.items()
    }


def _response_risk(
    *,
    logits_plus: float,
    logits_minus: float,
) -> torch.Tensor:
    values = _slice(_loss_inputs(), 0)
    response = values["label_increment"].to(dtype=torch.bool)
    with torch.no_grad():
        values["logits_plus"].zero_()
        values["logits_minus"].zero_()
        values["logits_plus"][response] = logits_plus
        values["logits_minus"][response] = logits_minus
    result = PairedEndpointCrossingLoss(LossConfig())(**values)
    return result["response_stratum_loss"]


def test_peco_response_formula_and_frozen_hierarchy_are_exact() -> None:
    config = LossConfig(dice_weight=0.7, epsilon=1.0e-5)
    values = _slice(_loss_inputs(), 0)
    result = PairedEndpointCrossingLoss(config)(**values)

    response = values["label_increment"].to(dtype=torch.bool)
    local = values["intervention_footprint"] & ~response
    global_context = (
        values["image_valid_mask"]
        & ~response
        & ~values["intervention_footprint"]
    )
    expected_response_field = 0.5 * (
        F.softplus(values["logits_plus"])
        + F.softplus(-values["logits_minus"])
    )
    expected_response = expected_response_field[response].mean()

    delta = (
        torch.sigmoid(values["logits_minus"])
        - torch.sigmoid(values["logits_plus"])
    )
    expected_local = delta.square()[local].mean()
    expected_global = delta.square()[global_context].mean()
    expected_zero = 0.5 * (expected_local + expected_global)
    expected_transition = 0.5 * (
        expected_response + expected_zero
    )

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
    expected_total = 0.5 * anchor["total"] + 0.5 * expected_transition

    torch.testing.assert_close(
        result["per_pair_response_stratum"][0],
        expected_response,
    )
    torch.testing.assert_close(
        result["per_pair_local_zero_stratum"][0],
        expected_local,
    )
    torch.testing.assert_close(
        result["per_pair_global_zero_stratum"][0],
        expected_global,
    )
    torch.testing.assert_close(result["per_pair_zero_risk"][0], expected_zero)
    torch.testing.assert_close(
        result["per_pair_transition"][0],
        expected_transition,
    )
    torch.testing.assert_close(
        result["per_pair_plus_anchor"][0],
        anchor["total"],
    )
    torch.testing.assert_close(result["total"], expected_total)
    torch.testing.assert_close(result["loss"], expected_total)
    assert result["zero_active_strata_per_pair"].tolist() == [2]
    assert result["transition_active_groups_per_pair"].tolist() == [2]


def test_response_gradient_moves_both_endpoints_across_zero() -> None:
    values = _slice(_loss_inputs(), 0)
    result = PairedEndpointCrossingLoss(LossConfig())(**values)
    plus_gradient, minus_gradient = torch.autograd.grad(
        result["total"],
        (values["logits_plus"], values["logits_minus"]),
    )
    response = values["label_increment"].to(dtype=torch.bool)

    # Gradient descent subtracts these gradients: plus moves down and minus up.
    assert torch.all(plus_gradient[response] > 0.0)
    assert torch.all(minus_gradient[response] < 0.0)
    assert torch.isfinite(plus_gradient).all()
    assert torch.isfinite(minus_gradient).all()


def test_both_high_both_low_and_wrong_endpoint_counterexamples() -> None:
    magnitude = 8.0
    correct = _response_risk(
        logits_plus=-magnitude,
        logits_minus=magnitude,
    )
    both_high = _response_risk(
        logits_plus=magnitude,
        logits_minus=magnitude,
    )
    both_low = _response_risk(
        logits_plus=-magnitude,
        logits_minus=-magnitude,
    )
    wrong = _response_risk(
        logits_plus=magnitude,
        logits_minus=-magnitude,
    )

    expected_correct = F.softplus(torch.tensor(-magnitude))
    expected_same_side = 0.5 * (
        F.softplus(torch.tensor(magnitude))
        + F.softplus(torch.tensor(-magnitude))
    )
    expected_wrong = F.softplus(torch.tensor(magnitude))
    torch.testing.assert_close(correct, expected_correct)
    torch.testing.assert_close(both_high, expected_same_side)
    torch.testing.assert_close(both_low, expected_same_side)
    torch.testing.assert_close(wrong, expected_wrong)
    correct_value = correct.detach().item()
    both_high_value = both_high.detach().item()
    both_low_value = both_low.detach().item()
    wrong_value = wrong.detach().item()
    assert correct_value < both_high_value
    assert correct_value < both_low_value
    assert both_high_value < wrong_value
    assert both_low_value < wrong_value


def test_empty_d_is_safe_and_pointwise_identical_to_parent() -> None:
    values = _slice(_loss_inputs(), 1)
    parent = OutcomeCompleteTransitionLoss(LossConfig())(**values)
    peco = PairedEndpointCrossingLoss(LossConfig())(**values)

    assert parent.keys() == peco.keys()
    for name in parent:
        assert torch.equal(peco[name], parent[name]), name
    assert torch.isfinite(peco["total"])
    assert peco["response_pixels_per_pair"].tolist() == [0]
    assert peco["response_active_per_pair"].tolist() == [False]
    assert peco["transition_active_groups_per_pair"].tolist() == [1]


def test_every_nonresponse_quantity_is_identical_to_parent() -> None:
    values = _loss_inputs()
    parent = OutcomeCompleteTransitionLoss(LossConfig())(**values)
    peco = PairedEndpointCrossingLoss(LossConfig())(**values)
    response_dependent = {
        "total",
        "loss",
        "transition_loss",
        "response_stratum_loss",
        "per_pair_total",
        "per_pair_transition",
        "per_pair_response_stratum",
    }

    assert parent.keys() == peco.keys()
    for name in parent.keys() - response_dependent:
        assert torch.equal(peco[name], parent[name]), name

    parent_zero_gradient = torch.autograd.grad(
        parent["zero_risk"],
        (values["logits_plus"], values["logits_minus"]),
        retain_graph=True,
    )
    peco_zero_gradient = torch.autograd.grad(
        peco["zero_risk"],
        (values["logits_plus"], values["logits_minus"]),
    )
    for actual, expected in zip(
        peco_zero_gradient,
        parent_zero_gradient,
        strict=True,
    ):
        assert torch.equal(actual, expected)


def test_loss_has_no_pair_kind_input_and_is_step_compatible_subclass() -> None:
    assert issubclass(
        PairedEndpointCrossingLoss,
        OutcomeCompleteTransitionLoss,
    )
    assert set(inspect.signature(PairedEndpointCrossingLoss).parameters) == {
        "config"
    }
    forward_parameters = set(
        inspect.signature(PairedEndpointCrossingLoss.forward).parameters
    )
    assert "pair_kind" not in forward_parameters
    assert "pair_kinds" not in forward_parameters


class _RecordingConservativeDecoder(
    CURELiteConservativeFactorizedDecoder
):
    def __init__(self) -> None:
        super().__init__(feature_channels=8, feature_stride=4)
        self.forward_inputs: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []
        self.forward_outputs: list[torch.Tensor] = []

    def forward(
        self,
        feature: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_inputs.append(
            (
                feature.detach().clone(),
                occupancy.detach().clone(),
            )
        )
        output = super().forward(feature, occupancy)
        output.retain_grad()
        self.forward_outputs.append(output)
        return output


def test_original_step_uses_one_2b_forward_and_preserves_gradients() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10101)
        outcome, factual = build_conservative_toy_case(
            SUPPORT_FAMILY,
            ((1, 6),),
        )
        outcome.pair_batch.feature.requires_grad_()
        for batch in factual.values():
            batch.feature.requires_grad_()

        decoder = _RecordingConservativeDecoder()
        parameters = tuple(decoder.parameters())
        assert len(parameters) == 6
        optimizer = torch.optim.Adam(decoder.parameters(), lr=1.0e-3)
        logs = outcome_complete_train_step(
            decoder,
            CURELiteLoss(),
            PairedEndpointCrossingLoss(LossConfig()),
            optimizer,
            factual,
            outcome,
        )

    assert logs["decoder_forward_calls_per_update"] == 3
    assert logs["decoder_states_per_update"] == 12
    assert logs["outcome/pairs"] == 2
    assert logs["outcome/endpoints"] == 4
    assert [int(value[0].shape[0]) for value in decoder.forward_inputs] == [
        4,
        4,
        4,
    ]
    pair_feature, pair_occupancy = decoder.forward_inputs[2]
    torch.testing.assert_close(
        pair_feature,
        torch.cat(
            (
                outcome.pair_batch.feature.detach(),
                outcome.pair_batch.feature.detach(),
            ),
            dim=0,
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        pair_occupancy,
        torch.cat(
            (
                outcome.pair_batch.occupancy_plus,
                outcome.pair_batch.occupancy_minus,
            ),
            dim=0,
        ),
    )

    endpoint_gradient = decoder.forward_outputs[2].grad
    assert endpoint_gradient is not None
    assert torch.isfinite(endpoint_gradient).all()
    batch_size = int(outcome.pair_batch.feature.shape[0])
    assert torch.count_nonzero(endpoint_gradient[:batch_size]) > 0
    assert torch.count_nonzero(endpoint_gradient[batch_size:]) > 0

    assert all(parameter.grad is not None for parameter in parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in parameters
        if parameter.grad is not None
    )
    assert outcome.pair_batch.feature.grad is None
    assert all(batch.feature.grad is None for batch in factual.values())
