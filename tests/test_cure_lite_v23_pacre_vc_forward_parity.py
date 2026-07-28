from __future__ import annotations

from dataclasses import fields as dataclass_fields

import pytest
import torch
from torch import Tensor

from cure_lite.experiment.coverage_state_paet_dr_gate import (
    _deterministic_execution_scope,
)
from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
    CoverageStatePACREFields,
)
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)


def _assert_raw_tensor_equal(first: Tensor, second: Tensor) -> None:
    assert first.shape == second.shape
    assert first.dtype == second.dtype
    assert first.device == second.device
    first_bits = (
        first.detach().contiguous().cpu().view(torch.uint8)
    )
    second_bits = (
        second.detach().contiguous().cpu().view(torch.uint8)
    )
    assert torch.equal(first_bits, second_bits)


def _models(
    *,
    seed: int,
    device: torch.device,
) -> tuple[
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CURELitePACREVerifierCorrectedLevelSet,
]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        v22 = (
            CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
                CoverageStatePACREConfig(
                    feature_channels=2,
                    feature_stride=2,
                    width=4,
                )
            )
        )
        torch.random.default_generator.manual_seed(seed)
        v23 = CURELitePACREVerifierCorrectedLevelSet(
            CoverageStatePACREVerifierCorrectedConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )
    return v22.to(device), v23.to(device)


def _inputs(
    *,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    feature = torch.randn(
        (1, 2, 3, 4),
        generator=generator,
        dtype=torch.float32,
    )
    occupancy = (
        torch.rand(
            (1, 1, 6, 8),
            generator=generator,
            dtype=torch.float32,
        )
        > 0.68
    )
    return (
        feature.contiguous().to(device),
        occupancy.contiguous().to(device),
    )


def _assert_state_dict_raw_equal(
    v22: torch.nn.Module,
    v23: torch.nn.Module,
) -> None:
    first = v22.state_dict()
    second = v23.state_dict()
    assert tuple(first) == tuple(second)
    for name in first:
        _assert_raw_tensor_equal(first[name], second[name])


def _assert_fields_raw_equal(
    first: CoverageStatePACREFields,
    second: CoverageStatePACREFields,
) -> None:
    assert type(first) is CoverageStatePACREFields
    assert type(second) is CoverageStatePACREFields
    for field in dataclass_fields(CoverageStatePACREFields):
        first_value = getattr(first, field.name)
        second_value = getattr(second, field.name)
        if isinstance(first_value, Tensor):
            assert isinstance(second_value, Tensor)
            _assert_raw_tensor_equal(first_value, second_value)
        else:
            assert first_value == second_value


def _probe_loss(fields: CoverageStatePACREFields) -> Tensor:
    return (
        fields.field.square().mean()
        + fields.actual_compatibility_hidden.square().mean()
        + fields.flipped_compatibility_hidden.square().mean()
    )


def _run_parity(*, device: torch.device, seed: int) -> None:
    v22, v23 = _models(seed=seed, device=device)
    feature, occupancy = _inputs(seed=seed + 1000, device=device)

    _assert_state_dict_raw_equal(v22, v23)
    with _deterministic_execution_scope():
        first_fields = v22.forward_fields(feature, occupancy)
        second_fields = v23.forward_fields(feature, occupancy)
        _assert_fields_raw_equal(first_fields, second_fields)

        first_parameters = dict(v22.named_parameters())
        second_parameters = dict(v23.named_parameters())
        assert tuple(first_parameters) == tuple(second_parameters)
        first_gradients = dict(
            zip(
                first_parameters,
                torch.autograd.grad(
                    _probe_loss(first_fields),
                    tuple(first_parameters.values()),
                    allow_unused=False,
                ),
                strict=True,
            )
        )
        second_gradients = dict(
            zip(
                second_parameters,
                torch.autograd.grad(
                    _probe_loss(second_fields),
                    tuple(second_parameters.values()),
                    allow_unused=False,
                ),
                strict=True,
            )
        )
        assert tuple(first_gradients) == tuple(second_gradients)
        for name in first_gradients:
            _assert_raw_tensor_equal(
                first_gradients[name],
                second_gradients[name],
            )


@pytest.mark.parametrize("seed", (42, 43, 44))
def test_cpu_all_fields_and_probe_gradients_are_raw_bit_exact(
    seed: int,
) -> None:
    _run_parity(device=torch.device("cpu"), seed=seed)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_small_cuda_all_fields_and_probe_gradients_are_raw_bit_exact() -> None:
    _run_parity(device=torch.device("cuda", 0), seed=42)
