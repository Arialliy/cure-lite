from __future__ import annotations

import pytest
import torch

from cure_lite.conservative_factorized_config import (
    ConservativeFactorizedDecoderConfig,
)
from cure_lite.conservative_factorized_decoder import (
    CURELiteConservativeFactorizedDecoder,
    coverage_conserving_phase_evidence,
)
from cure_lite.factorized_config import FactorizedDecoderConfig
from cure_lite.factorized_decoder import CURELiteFactorizedDecoder


def _decoder(
    *,
    channels: int = 3,
    stride: int = 4,
    seed: int = 8117,
) -> CURELiteConservativeFactorizedDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CURELiteConservativeFactorizedDecoder(
            ConservativeFactorizedDecoderConfig(
                feature_channels=channels,
                feature_stride=stride,
            )
        )


def test_operator_conserves_one_budget_and_competes_over_phases() -> None:
    raw = torch.tensor(
        [[[[4.0]], [[0.0]], [[-1.0]], [[-2.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    burden = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    (
        common_mode,
        margin,
        budget,
        allocation,
        evidence,
    ) = coverage_conserving_phase_evidence(raw, burden)

    assert torch.equal(common_mode, margin)
    assert torch.all(evidence >= 0.0)
    torch.testing.assert_close(
        allocation.sum(dim=1, keepdim=True),
        torch.ones_like(budget),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        evidence.sum(dim=1, keepdim=True),
        budget,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert float(allocation[0, 0].detach()) > 0.95
    assert float(evidence[0, 0].detach()) > float(
        evidence[0, 1].detach()
    )

    evidence.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert torch.all(raw.grad > 0.0)


def test_uniform_raw_evidence_has_uniform_allocation_and_zero_anchor() -> None:
    raw = torch.zeros((2, 16, 3, 5), dtype=torch.float32)
    burden = torch.zeros((2, 1, 3, 5), dtype=torch.float32)
    (
        common_mode,
        margin,
        budget,
        allocation,
        evidence,
    ) = coverage_conserving_phase_evidence(raw, burden)

    assert torch.equal(common_mode, torch.zeros_like(common_mode))
    assert torch.equal(margin, torch.zeros_like(margin))
    assert torch.equal(budget, torch.zeros_like(budget))
    assert torch.equal(evidence, torch.zeros_like(evidence))
    torch.testing.assert_close(
        allocation,
        torch.full_like(allocation, 1.0 / 16.0),
        rtol=0.0,
        atol=0.0,
    )


def test_common_mode_and_phase_contrast_are_orthogonal() -> None:
    raw = torch.tensor(
        [[[[1.0]], [[-1.0]], [[0.5]], [[-0.5]]]],
        dtype=torch.float64,
    )
    burden = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    base = coverage_conserving_phase_evidence(raw, burden)
    shifted = coverage_conserving_phase_evidence(raw + 2.0, burden)
    contrast_delta = torch.tensor(
        [[[[1.5]], [[-1.5]], [[0.5]], [[-0.5]]]],
        dtype=torch.float64,
    )
    contrasted = coverage_conserving_phase_evidence(
        raw + contrast_delta,
        burden,
    )

    torch.testing.assert_close(
        shifted[0],
        base[0] + 2.0,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        shifted[3],
        base[3],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        contrasted[0],
        base[0],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(contrasted[3], base[3])


def test_occupancy_release_changes_budget_not_allocation() -> None:
    raw = torch.tensor(
        [[[[2.0]], [[1.0]], [[0.0]], [[-1.0]]]],
        dtype=torch.float64,
    )
    plus_burden = torch.full(
        (1, 1, 1, 1),
        torch.log(torch.tensor(2.0, dtype=torch.float64)),
        dtype=torch.float64,
    )
    minus_burden = torch.zeros_like(plus_burden)
    plus = coverage_conserving_phase_evidence(raw, plus_burden)
    minus = coverage_conserving_phase_evidence(raw, minus_burden)

    torch.testing.assert_close(
        plus[3],
        minus[3],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.all(minus[2] >= plus[2])
    assert torch.all(minus[4] >= plus[4])
    assert torch.any(minus[4] > plus[4])


def test_decoder_keeps_exact_topology_state_and_parameter_budget() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2027)
        v4 = CURELiteFactorizedDecoder(
            FactorizedDecoderConfig(64, 4)
        )
        torch.manual_seed(2027)
        v8 = CURELiteConservativeFactorizedDecoder(
            ConservativeFactorizedDecoderConfig(64, 4)
        )

    assert tuple(v4.state_dict()) == tuple(v8.state_dict())
    for name, expected in v4.state_dict().items():
        assert torch.equal(v8.state_dict()[name], expected)
    assert sum(parameter.numel() for parameter in v8.parameters()) == 4385
    assert len(tuple(v8.parameters())) == 6
    assert tuple(type(module) for module in tuple(v8.modules())[1:]) == tuple(
        type(module) for module in tuple(v4.modules())[1:]
    )


def test_decoder_fields_conserve_budget_before_pixel_shuffle() -> None:
    decoder = _decoder(channels=3, stride=4)
    feature = torch.randn(2, 3, 5, 7)
    occupancy = torch.zeros(2, 1, 20, 28, dtype=torch.bool)
    occupancy[0, 0, 9, 13] = True

    fields = decoder.forward_fields(feature, occupancy)

    assert fields.raw_phase_evidence.shape == (2, 16, 5, 7)
    assert fields.evidence_budget.shape == (2, 1, 5, 7)
    assert fields.evidence.shape == (2, 1, 20, 28)
    torch.testing.assert_close(
        fields.phase_allocation.sum(dim=1, keepdim=True),
        torch.ones_like(fields.evidence_budget),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        fields.allocated_phase_evidence.sum(
            dim=1,
            keepdim=True,
        ),
        fields.evidence_budget,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert torch.equal(
        fields.evidence,
        decoder.pixel_shuffle(fields.allocated_phase_evidence),
    )


def test_identity_and_count_support_locality_are_exact() -> None:
    decoder = _decoder(channels=2, stride=2)
    feature = torch.randn(1, 2, 4, 4)
    plus = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    plus[0, 0, 2, 2] = True
    minus = plus.clone()
    minus[0, 0, 2, 2] = False

    identity_a = decoder(feature, plus)
    identity_b = decoder(feature, plus.clone())
    assert torch.equal(identity_a, identity_b)

    plus_fields = decoder.forward_fields(feature, plus)
    minus_fields = decoder.forward_fields(feature, minus)
    count_release = (
        plus_fields.local_occupancy_count
        - minus_fields.local_occupancy_count
    )
    native_support = count_release > 0.0
    output_support = torch.nn.functional.pixel_shuffle(
        native_support.expand(-1, 4, -1, -1).to(torch.float32),
        2,
    ).to(torch.bool)
    delta = minus_fields.logits - plus_fields.logits

    assert torch.all(delta >= 0.0)
    assert torch.equal(
        delta[~output_support],
        torch.zeros_like(delta[~output_support]),
    )


def test_zero_feature_has_no_occupancy_response() -> None:
    decoder = _decoder(channels=3, stride=4)
    feature = torch.zeros(1, 3, 3, 3)
    plus = torch.zeros(1, 1, 12, 12, dtype=torch.bool)
    plus[0, 0, 4, 4] = True
    minus = torch.zeros_like(plus)

    plus_fields = decoder.forward_fields(feature, plus)
    minus_fields = decoder.forward_fields(feature, minus)

    assert torch.equal(
        plus_fields.evidence,
        torch.zeros_like(plus_fields.evidence),
    )
    assert torch.equal(
        minus_fields.evidence,
        torch.zeros_like(minus_fields.evidence),
    )
    assert torch.equal(plus_fields.logits, minus_fields.logits)


def test_all_six_parameter_tensors_receive_finite_gradients() -> None:
    decoder = _decoder(channels=3, stride=2)
    feature = torch.randn(2, 3, 5, 5)
    plus = torch.zeros(2, 1, 10, 10, dtype=torch.bool)
    minus = torch.zeros_like(plus)
    plus[:, :, 4:6, 4:6] = True

    delta = torch.sigmoid(decoder(feature, minus)) - torch.sigmoid(
        decoder(feature, plus)
    )
    loss = (delta - 0.5).square().mean()
    loss.backward()

    parameters = tuple(decoder.named_parameters())
    assert len(parameters) == 6
    for _, parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "invalid",
    [
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
    ],
)
def test_operator_rejects_invalid_raw_fields(
    invalid: torch.Tensor,
) -> None:
    burden = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    with pytest.raises((TypeError, ValueError)):
        coverage_conserving_phase_evidence(
            invalid.reshape(1, -1, 1, 1),
            burden,
        )
