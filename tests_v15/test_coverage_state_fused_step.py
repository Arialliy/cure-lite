from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from cure_lite.coverage_state_batches import CoverageStateFusedBatch
from cure_lite.coverage_state_level_set import (
    CURELiteCoverageStateLevelSet,
    CoverageStateLevelSetConfig,
)
from cure_lite.coverage_state_sobolev import (
    CSLF_COMPLETION_ROOTED_RESPONSE_POLICY,
    CSLF_PMOPE_POLICY,
    CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
    CoverageStateSobolevConfig,
)
from cure_lite.coverage_state_supremal_projection import (
    CSLF_USCOPE_POLICY,
)
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
    coverage_state_fused_train_step,
)
from tests_v15.coverage_state_test_helpers import TOY_STRIDE
from tests_v15.coverage_state_training_test_helpers import (
    make_training_fused_batch,
)


class CountingCSLF(CURELiteCoverageStateLevelSet):
    def __init__(self) -> None:
        super().__init__(
            CoverageStateLevelSetConfig(
                feature_channels=2,
                feature_stride=TOY_STRIDE,
                width=4,
            )
        )
        self.forward_calls = 0

    def forward(
        self,
        feature: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(feature, occupancy)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, lr=1.0e-2)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def _config() -> CoverageStateSobolevConfig:
    return CoverageStateSobolevConfig(truncation_radius=TOY_STRIDE)


@pytest.mark.parametrize(
    "objective",
    tuple(CoverageStatePairObjective),
)
def test_fused_step_uses_one_forward_backward_step_and_twelve_states(
    objective: CoverageStatePairObjective,
) -> None:
    torch.manual_seed(7)
    model = CountingCSLF()
    optimizer = CountingSGD(model.parameters())
    logs = coverage_state_fused_train_step(
        model,
        optimizer,
        make_training_fused_batch(),
        config=_config(),
        pair_objective=objective,
    )
    assert model.forward_calls == 1
    assert optimizer.step_calls == 1
    assert logs["model_forward_calls"] == 1
    assert logs["backward_calls"] == 1
    assert logs["optimizer_steps"] == 1
    assert logs["logical_states"] == 12
    assert logs["factual_miss_states"] == 4
    assert logs["factual_no_miss_states"] == 4
    assert logs["pair_count"] == 2
    assert logs["pair_endpoint_states"] == 4
    assert logs["identity_null_optimizer_exposure"] == 0
    assert logs["diagnostic_only_optimizer_exposure"] == 0
    assert logs["pair_objective"] == objective.value
    expected_policy = {
        CoverageStatePairObjective.COMPLETION_ROOTED_RESPONSE_JOINT: (
            CSLF_COMPLETION_ROOTED_RESPONSE_POLICY
        ),
        CoverageStatePairObjective.SUPPORT_ORIENTED_RESPONSE_JOINT: (
            CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
        ),
        CoverageStatePairObjective.PMOPE_JOINT: CSLF_PMOPE_POLICY,
        CoverageStatePairObjective.USCOPE_JOINT: CSLF_USCOPE_POLICY,
    }.get(objective, objective.value)
    assert logs["pair_objective_policy"] == expected_policy
    assert logs["total"] == pytest.approx(
        logs["factual_miss/loss"]
        + logs["factual_no_miss/loss"]
        + logs["pair/loss"],
        abs=1.0e-6,
    )
    assert logs["gradient_l2_norm"] > 0.0


def test_three_objectives_share_exact_initial_state_and_selection() -> None:
    torch.manual_seed(19)
    initial = CountingCSLF()
    initial_state = deepcopy(initial.state_dict())
    fingerprints: set[str] = set()
    for objective in CoverageStatePairObjective:
        model = CountingCSLF()
        model.load_state_dict(initial_state, strict=True)
        before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        assert all(
            torch.equal(value, before[name])
            for name, value in model.state_dict().items()
        )
        logs = coverage_state_fused_train_step(
            model,
            CountingSGD(model.parameters()),
            make_training_fused_batch(),
            config=_config(),
            pair_objective=objective,
        )
        fingerprints.add(str(logs["selection_fingerprint"]))
    assert len(fingerprints) == 1


def test_first_update_allows_zero_trunk_gradient_then_recovers() -> None:
    torch.manual_seed(23)
    model = CountingCSLF()
    optimizer = CountingSGD(model.parameters())
    batch = make_training_fused_batch()
    coverage_state_fused_train_step(
        model,
        optimizer,
        batch,
        config=_config(),
        pair_objective=CoverageStatePairObjective.RESPONSE_JOINT,
    )
    assert model.input_projection.weight.grad is not None
    assert model.spatial_mixing.weight.grad is not None
    assert not bool(torch.any(model.input_projection.weight.grad != 0.0))
    assert not bool(torch.any(model.spatial_mixing.weight.grad != 0.0))
    assert bool(torch.any(model.phase_projection.weight.grad != 0.0))

    coverage_state_fused_train_step(
        model,
        optimizer,
        batch,
        config=_config(),
        pair_objective=CoverageStatePairObjective.RESPONSE_JOINT,
    )
    assert bool(torch.any(model.input_projection.weight.grad != 0.0))
    assert bool(torch.any(model.spatial_mixing.weight.grad != 0.0))


def test_fused_step_accepts_a_finite_zero_gradient_at_convergence() -> None:
    torch.manual_seed(29)
    model = CountingCSLF()
    optimizer = CountingSGD(model.parameters())
    before = deepcopy(model.state_dict())
    handles = [
        parameter.register_hook(torch.zeros_like)
        for parameter in model.parameters()
    ]
    try:
        logs = coverage_state_fused_train_step(
            model,
            optimizer,
            make_training_fused_batch(),
            config=_config(),
            pair_objective=CoverageStatePairObjective.RESPONSE_JOINT,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert logs["gradient_l2_norm"] == 0.0
    assert logs["nonzero_gradient_parameter_count"] == 0
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )


def test_fused_step_materializes_model_inputs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = CoverageStateFusedBatch.model_inputs

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(CoverageStateFusedBatch, "model_inputs", counted)
    model = CountingCSLF()
    coverage_state_fused_train_step(
        model,
        CountingSGD(model.parameters()),
        make_training_fused_batch(),
        config=_config(),
        pair_objective=CoverageStatePairObjective.RESPONSE_JOINT,
    )
    assert calls == 1


def test_separable_step_does_not_call_joint_pair_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cure_lite.train.coverage_state_fused_step as module

    def forbidden(*args, **kwargs):
        raise AssertionError("separable endpoint consumed a joint pair loss")

    monkeypatch.setattr(
        module,
        "coverage_state_pair_sobolev_loss_from_targets",
        forbidden,
    )
    monkeypatch.setattr(
        module,
        "coverage_state_identity_joint_loss_from_targets",
        forbidden,
    )
    model = CountingCSLF()
    logs = coverage_state_fused_train_step(
        model,
        CountingSGD(model.parameters()),
        make_training_fused_batch(),
        config=_config(),
        pair_objective=CoverageStatePairObjective.SEPARABLE_ENDPOINT,
    )
    assert logs["pair_objective"] == "separable_endpoint"


def test_invalid_optimizer_fails_before_model_or_gradient_mutation() -> None:
    model = CountingCSLF()
    model.eval()
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(parameters[:-1], lr=1.0e-2)
    snapshot = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    with pytest.raises(ValueError, match="every trainable FP32"):
        coverage_state_fused_train_step(
            model,
            optimizer,
            make_training_fused_batch(),
            config=_config(),
            pair_objective=CoverageStatePairObjective.RESPONSE_JOINT,
        )
    assert not model.training
    assert model.forward_calls == 0
    assert all(parameter.grad is None for parameter in parameters)
    assert all(
        torch.equal(value, snapshot[name])
        for name, value in model.state_dict().items()
    )


def test_unknown_objective_fails_before_forward() -> None:
    model = CountingCSLF()
    with pytest.raises(ValueError, match="unknown coverage-state"):
        coverage_state_fused_train_step(
            model,
            CountingSGD(model.parameters()),
            make_training_fused_batch(),
            config=_config(),
            pair_objective="not-an-objective",
        )
    assert model.forward_calls == 0
