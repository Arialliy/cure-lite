from __future__ import annotations

import pytest
import torch
from torch import nn

from cure_lite.train.step import BranchBatch, _validate_optimizer_scope


def _decoder() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 4), nn.SiLU(), nn.Linear(4, 1))


def test_optimizer_scope_accepts_every_decoder_parameter_exactly_once() -> None:
    decoder = _decoder()
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)

    _validate_optimizer_scope(decoder, optimizer)


def test_optimizer_scope_rejects_a_missing_decoder_parameter() -> None:
    decoder = _decoder()
    parameters = list(decoder.parameters())
    optimizer = torch.optim.SGD(parameters[:-1], lr=0.1)

    with pytest.raises(ValueError, match="every decoder parameter"):
        _validate_optimizer_scope(decoder, optimizer)


def test_optimizer_scope_rejects_duplicate_decoder_parameters() -> None:
    decoder = _decoder()
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)
    optimizer.param_groups[0]["params"].append(optimizer.param_groups[0]["params"][0])

    with pytest.raises(ValueError, match="duplicate decoder parameters"):
        _validate_optimizer_scope(decoder, optimizer)


def test_optimizer_scope_rejects_frozen_decoder_parameters() -> None:
    decoder = _decoder()
    list(decoder.parameters())[0].requires_grad_(False)
    optimizer = torch.optim.SGD(
        [parameter for parameter in decoder.parameters() if parameter.requires_grad],
        lr=0.1,
    )

    with pytest.raises(ValueError, match="must require gradients"):
        _validate_optimizer_scope(decoder, optimizer)


def _branch_batch(*, positive: bool) -> BranchBatch:
    occupancy = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    target = torch.zeros(1, 1, 3, 3)
    if positive:
        target[0, 0, 1, 1] = 1.0
    return BranchBatch(
        feature=torch.zeros(1, 2, 3, 3),
        occupancy=occupancy,
        target=target,
        valid_mask=torch.ones_like(occupancy),
    )


def test_branch_batch_rejects_positive_no_miss_state() -> None:
    with pytest.raises(ValueError, match="empty target"):
        _branch_batch(positive=True).validate(expected_branch="factual_no_miss")


def test_no_miss_slot_rejects_a_diagnostics_only_unreachable_state() -> None:
    batch = _branch_batch(positive=False)
    diagnostics_only = BranchBatch(
        feature=batch.feature,
        occupancy=batch.occupancy,
        target=batch.target,
        valid_mask=torch.zeros_like(batch.valid_mask),
    )

    with pytest.raises(ValueError, match="valid negative supervision"):
        diagnostics_only.validate(expected_branch="factual_no_miss")


@pytest.mark.parametrize("branch", ["factual_miss", "synthetic"])
def test_positive_branches_reject_an_empty_state(branch: str) -> None:
    with pytest.raises(ValueError, match="positive supervision"):
        _branch_batch(positive=False).validate(expected_branch=branch)
