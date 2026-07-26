from __future__ import annotations

import pytest
import torch

from cure_lite.losses import CURELiteLoss
from cure_lite.paired_control_losses import (
    after_only_absolute_synthetic_loss,
    build_geometry_matched_endpoint_supervision,
    geometry_matched_independent_endpoint_loss,
    minus_detached_paired_difference_loss,
    plus_detached_paired_difference_loss,
)
from cure_lite.paired_losses import PairedDifferenceLoss


def _control_geometry() -> dict[str, torch.Tensor]:
    # Pixel roles:
    # 0: writable background
    # 1,2: pre-existing unmatched target, present in R(O+) and R(O-)
    # 3,4: selected target, newly writable in R(O-)
    # 5: occupied background
    # 6: invalid image pixel
    valid = torch.tensor([[[[1, 1, 1, 1, 1, 1, 0]]]], dtype=torch.bool)
    gt_union = torch.tensor([[[[0, 1, 1, 1, 1, 0, 0]]]], dtype=torch.bool)
    occupancy_plus = torch.tensor(
        [[[[0, 0, 0, 1, 0, 1, 0]]]],
        dtype=torch.bool,
    )
    occupancy_minus = torch.tensor(
        [[[[0, 0, 0, 0, 0, 1, 0]]]],
        dtype=torch.bool,
    )
    completion_plus = torch.tensor(
        [[[[0, 1, 1, 0, 0, 0, 0]]]],
        dtype=torch.bool,
    )
    completion_minus = torch.tensor(
        [[[[0, 1, 1, 1, 1, 0, 0]]]],
        dtype=torch.bool,
    )
    selected_completion = completion_minus & ~completion_plus
    return {
        "valid": valid,
        "gt_union": gt_union,
        "occupancy_plus": occupancy_plus,
        "occupancy_minus": occupancy_minus,
        "completion_plus": completion_plus,
        "completion_minus": completion_minus,
        "selected_completion": selected_completion,
    }


def test_geometry_matched_endpoint_has_exact_T_B_M_pixel_semantics() -> None:
    state = _control_geometry()

    plus = build_geometry_matched_endpoint_supervision(
        state["completion_plus"],
        state["occupancy_plus"],
        state["gt_union"],
        state["valid"],
    )
    minus = build_geometry_matched_endpoint_supervision(
        state["completion_minus"],
        state["occupancy_minus"],
        state["gt_union"],
        state["valid"],
    )

    expected_background = torch.tensor(
        [[[[1, 0, 0, 0, 0, 0, 0]]]],
        dtype=torch.bool,
    )
    assert torch.equal(
        plus["target"],
        state["completion_plus"].to(torch.float32),
    )
    assert torch.equal(
        minus["target"],
        state["completion_minus"].to(torch.float32),
    )
    assert torch.equal(plus["background"], expected_background)
    assert torch.equal(minus["background"], expected_background)
    assert torch.equal(
        plus["valid_mask"],
        state["completion_plus"] | expected_background,
    )
    assert torch.equal(
        minus["valid_mask"],
        state["completion_minus"] | expected_background,
    )

    # T+ is the full completion field, not the new selected target A and not 0.
    assert torch.count_nonzero(plus["target"]) == 2
    assert not torch.equal(
        plus["target"].to(torch.bool),
        state["selected_completion"],
    )


def test_geometry_matched_control_rejects_nonwritable_completion_pixels() -> None:
    state = _control_geometry()
    invalid_completion = state["completion_plus"].clone()
    invalid_completion[..., 5] = True

    with pytest.raises(ValueError, match="valid and writable"):
        build_geometry_matched_endpoint_supervision(
            invalid_completion,
            state["occupancy_plus"],
            state["gt_union"],
            state["valid"],
        )


def test_independent_endpoint_erm_uses_endpoint_then_pair_macro_reduction() -> None:
    first = _control_geometry()
    second = {
        name: value.flip(-1)
        for name, value in first.items()
    }
    completion_plus = torch.cat(
        (first["completion_plus"], second["completion_plus"]),
        dim=0,
    )
    completion_minus = torch.cat(
        (first["completion_minus"], second["completion_minus"]),
        dim=0,
    )
    occupancy_plus = torch.cat(
        (first["occupancy_plus"], second["occupancy_plus"]),
        dim=0,
    )
    occupancy_minus = torch.cat(
        (first["occupancy_minus"], second["occupancy_minus"]),
        dim=0,
    )
    gt_union = torch.cat((first["gt_union"], second["gt_union"]), dim=0)
    valid = torch.cat((first["valid"], second["valid"]), dim=0)
    logits_plus = torch.tensor(
        [
            [[[0.1, -0.5, 0.8, 1.1, -0.2, 0.4, -1.0]]],
            [[[0.2, 0.9, -0.1, 0.5, -0.7, 1.2, -0.4]]],
        ]
    )
    logits_minus = torch.tensor(
        [
            [[[-0.3, 0.4, 1.0, -0.9, 0.7, 0.2, -0.6]]],
            [[[0.6, -0.8, 0.3, 1.4, -0.2, 0.1, -1.1]]],
        ]
    )
    criterion = CURELiteLoss()

    result = geometry_matched_independent_endpoint_loss(
        logits_plus,
        logits_minus,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        gt_union=gt_union,
        image_valid_mask=valid,
        criterion=criterion,
    )

    manual_pair_losses = []
    for index in range(2):
        plus = criterion(
            logits_plus[index],
            result["target_plus"][index],
            result["valid_mask_plus"][index],
        )["total"]
        minus = criterion(
            logits_minus[index],
            result["target_minus"][index],
            result["valid_mask_minus"][index],
        )["total"]
        manual_pair_losses.append(0.5 * (plus + minus))
    expected = torch.stack(manual_pair_losses)
    torch.testing.assert_close(result["per_pair_total"], expected)
    torch.testing.assert_close(result["total"], expected.mean())
    assert int(result["pair_count"]) == 2


def test_independent_endpoint_erm_is_separable_and_not_main_objective() -> None:
    state = _control_geometry()
    logits_plus = torch.tensor(
        [[[[0.2, -0.4, 0.8, -0.6, 0.3, 1.1, -0.2]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    logits_minus = torch.tensor(
        [[[[1.0, 0.3, -0.1, 0.7, -0.8, 0.4, -0.5]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    independent = geometry_matched_independent_endpoint_loss(
        logits_plus,
        logits_minus,
        completion_plus=state["completion_plus"],
        completion_minus=state["completion_minus"],
        occupancy_plus=state["occupancy_plus"],
        occupancy_minus=state["occupancy_minus"],
        gt_union=state["gt_union"],
        image_valid_mask=state["valid"],
    )["total"]
    grad_minus = torch.autograd.grad(
        independent,
        logits_minus,
        create_graph=True,
    )[0]
    mixed = torch.autograd.grad(
        grad_minus.sum(),
        logits_plus,
        allow_unused=True,
    )[0]

    main = PairedDifferenceLoss()(
        logits_plus,
        logits_minus,
        state["selected_completion"].to(torch.float32),
        state["valid"],
    )["total"]
    main_grad_minus = torch.autograd.grad(
        main,
        logits_minus,
        create_graph=True,
    )[0]
    main_mixed = torch.autograd.grad(
        main_grad_minus.sum(),
        logits_plus,
    )[0]

    assert mixed is None
    assert torch.all(main_mixed[state["valid"]] != 0.0)
    assert torch.all(main_mixed[~state["valid"]] == 0.0)
    assert not torch.isclose(independent, main)


def test_after_only_is_old_atomic_synthetic_loss_and_blocks_plus_gradient() -> None:
    state = _control_geometry()
    logits_plus = torch.tensor(
        [[[[0.2, -0.4, 0.8, -0.6, 0.3, 1.1, -0.2]]]],
        requires_grad=True,
    )
    logits_minus = torch.tensor(
        [[[[1.0, 0.3, -0.1, 0.7, -0.8, 0.4, -0.5]]]],
        requires_grad=True,
    )
    result = after_only_absolute_synthetic_loss(
        logits_plus,
        logits_minus,
        selected_completion=state["selected_completion"],
        occupancy_minus=state["occupancy_minus"],
        gt_union=state["gt_union"],
        image_valid_mask=state["valid"],
    )
    expected_valid = state["selected_completion"] | torch.tensor(
        [[[[1, 0, 0, 0, 0, 0, 0]]]],
        dtype=torch.bool,
    )
    expected = CURELiteLoss()(
        logits_minus,
        state["selected_completion"].to(torch.float32),
        expected_valid,
    )["total"]

    torch.testing.assert_close(result["total"], expected)
    assert torch.equal(result["valid_mask_minus"], expected_valid)
    assert not torch.any(
        result["valid_mask_minus"] & state["completion_plus"]
    )

    result["total"].backward()
    assert logits_plus.grad is None
    assert logits_minus.grad is not None
    assert torch.any(logits_minus.grad != 0.0)

    changed_plus = after_only_absolute_synthetic_loss(
        logits_plus.detach() + 100.0,
        logits_minus.detach(),
        selected_completion=state["selected_completion"],
        occupancy_minus=state["occupancy_minus"],
        gt_union=state["gt_union"],
        image_valid_mask=state["valid"],
    )["total"]
    torch.testing.assert_close(changed_plus, result["total"].detach())


@pytest.mark.parametrize(
    ("control", "detached_name"),
    [
        (plus_detached_paired_difference_loss, "plus"),
        (minus_detached_paired_difference_loss, "minus"),
    ],
)
def test_stop_gradient_controls_preserve_value_but_isolate_one_endpoint(
    control,
    detached_name: str,
) -> None:
    state = _control_geometry()
    logits_plus = torch.tensor(
        [[[[0.2, -0.4, 0.8, -0.6, 0.3, 1.1, -0.2]]]],
        requires_grad=True,
    )
    logits_minus = torch.tensor(
        [[[[1.0, 0.3, -0.1, 0.7, -0.8, 0.4, -0.5]]]],
        requires_grad=True,
    )
    label_increment = state["selected_completion"].to(torch.float32)
    main_value = PairedDifferenceLoss()(
        logits_plus,
        logits_minus,
        label_increment,
        state["valid"],
    )["total"]
    controlled = control(
        logits_plus,
        logits_minus,
        label_increment,
        state["valid"],
    )["total"]

    torch.testing.assert_close(controlled, main_value)
    controlled.backward()
    if detached_name == "plus":
        assert logits_plus.grad is None
        assert logits_minus.grad is not None
        assert torch.any(logits_minus.grad != 0.0)
    else:
        assert logits_plus.grad is not None
        assert torch.any(logits_plus.grad != 0.0)
        assert logits_minus.grad is None
