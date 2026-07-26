from __future__ import annotations

import pytest
import torch
from torch import nn

from cure_lite.nlcc_dataset_free_runner import audit_finite_training_state


class _BufferedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.register_buffer("scale", torch.ones(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value) * self.scale


def _initialized_state() -> tuple[_BufferedLinear, torch.optim.Adam]:
    model = _BufferedLinear()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 2)).sum()
    loss.backward()
    optimizer.step()
    return model, optimizer


def test_finite_state_audit_covers_parameters_buffers_and_adam_state() -> None:
    model = _BufferedLinear()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = audit_finite_training_state(
        model,
        optimizer,
        update_index=-1,
        phase="before_first_update",
    )
    assert initial["parameter_tensor_count"] == 2
    assert initial["buffer_tensor_count"] == 1
    assert initial["optimizer_state_tensor_count"] == 0
    assert initial["nonfinite_element_count"] == 0

    model, optimizer = _initialized_state()
    after = audit_finite_training_state(
        model,
        optimizer,
        update_index=0,
        phase="after_optimizer_step",
    )
    assert after["optimizer_state_tensor_count"] == 6
    assert after["nonfinite_element_count"] == 0
    assert after["all_finite"] is True


@pytest.mark.parametrize("location", ["parameter", "buffer", "optimizer_state"])
def test_finite_state_audit_rejects_every_nonfinite_state_location(
    location: str,
) -> None:
    model, optimizer = _initialized_state()
    if location == "parameter":
        next(model.parameters()).data.reshape(-1)[0] = float("nan")
    elif location == "buffer":
        model.scale[0] = float("inf")
    else:
        state = next(iter(optimizer.state.values()))
        state["exp_avg"].reshape(-1)[0] = float("-inf")

    with pytest.raises(FloatingPointError, match="non-finite training state"):
        audit_finite_training_state(
            model,
            optimizer,
            update_index=0,
            phase="after_optimizer_step",
        )


def test_finite_state_audit_rejects_unknown_phase() -> None:
    model, optimizer = _initialized_state()
    with pytest.raises(ValueError, match="unknown finite-state audit phase"):
        audit_finite_training_state(
            model,
            optimizer,
            update_index=0,
            phase="after_backward",
        )
