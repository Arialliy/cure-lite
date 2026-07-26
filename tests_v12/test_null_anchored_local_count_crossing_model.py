from __future__ import annotations

import torch

from cure_lite.factorized_model import CURELiteFactorizedModel
from cure_lite.null_anchored_local_count_crossing_decoder import (
    CURELiteNullAnchoredLocalCountCrossingDecoder,
)
from tests.test_crossing_factorized_model import _ContractAdapter


def test_frozen_factorized_model_accepts_nlcc_without_a_new_wrapper() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12012)
        adapter = _ContractAdapter(channels=2, stride=2)
        decoder = CURELiteNullAnchoredLocalCountCrossingDecoder(
            feature_channels=2,
            feature_stride=2,
        )
        model = CURELiteFactorizedModel(adapter, decoder)
        model.train()

        base_outputs: list[torch.Tensor] = []
        decoder_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        decoder_outputs: list[torch.Tensor] = []
        parameter_gradients: dict[str, list[torch.Tensor]] = {
            name: [] for name, _ in decoder.named_parameters()
        }

        def record_base_output(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            base_outputs.append(output)

        def record_decoder_input(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            feature, occupancy = inputs
            decoder_inputs.append((feature, occupancy))

        def record_decoder_output(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            decoder_outputs.append(output)

        def parameter_hook(name: str):
            def record(gradient: torch.Tensor) -> None:
                parameter_gradients[name].append(
                    gradient.detach().clone()
                )

            return record

        handles = [
            adapter.base.register_forward_hook(record_base_output),
            decoder.register_forward_pre_hook(record_decoder_input),
            decoder.register_forward_hook(record_decoder_output),
        ]
        handles.extend(
            parameter.register_hook(parameter_hook(name))
            for name, parameter in decoder.named_parameters()
        )
        images = torch.randn(2, 1, 8, 8, requires_grad=True)
        try:
            output = model(images, residual_threshold=0.0)
            weights = torch.linspace(
                0.5,
                1.5,
                output.residual_logits.numel(),
                dtype=output.residual_logits.dtype,
                device=output.residual_logits.device,
            ).reshape_as(output.residual_logits)
            (output.residual_logits * weights).mean().backward()
        finally:
            for handle in handles:
                handle.remove()

    assert type(model) is CURELiteFactorizedModel
    assert model.base is adapter
    assert model.decoder is decoder
    assert tuple(model._modules) == ("base", "decoder")

    assert adapter.extract_calls == 1
    assert adapter.base.forward_calls == 1
    assert len(base_outputs) == 1
    assert len(decoder_inputs) == 1
    assert len(decoder_outputs) == 1
    assert base_outputs[0].requires_grad is False

    decoder_feature, decoder_occupancy = decoder_inputs[0]
    assert decoder_feature.requires_grad is False
    assert decoder_feature.grad_fn is None
    assert torch.equal(decoder_occupancy, output.occupancy)
    assert torch.equal(decoder_outputs[0], output.residual_logits)

    expected_occupancy = (
        output.base_probability >= model.occupancy_threshold
    )
    assert torch.equal(output.occupancy, expected_occupancy)
    unmasked_probability = torch.sigmoid(decoder_outputs[0])
    assert torch.equal(
        output.residual_probability[~output.occupancy],
        unmasked_probability[~output.occupancy],
    )
    assert torch.all(unmasked_probability[output.occupancy] > 0.0)
    assert torch.equal(
        output.residual_probability[output.occupancy],
        torch.zeros_like(
            output.residual_probability[output.occupancy]
        ),
    )
    assert not torch.any(output.residual_mask & output.occupancy)
    assert torch.equal(
        output.final_mask,
        output.occupancy | output.residual_mask,
    )
    assert torch.all(output.final_mask[output.occupancy])

    assert adapter.training is False
    assert all(
        not parameter.requires_grad for parameter in adapter.parameters()
    )
    assert all(
        parameter.grad is None for parameter in adapter.parameters()
    )
    assert images.grad is None
    assert tuple(model.trainable_parameters()) == tuple(
        decoder.parameters()
    )
    assert len(parameter_gradients) == 6
    assert all(
        len(gradients) == 1
        and torch.isfinite(gradients[0]).all()
        and torch.count_nonzero(gradients[0]) > 0
        for gradients in parameter_gradients.values()
    )


def test_infer_is_identical_and_keeps_one_base_one_decoder_per_call() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12013)
        adapter = _ContractAdapter(channels=2, stride=2)
        decoder = CURELiteNullAnchoredLocalCountCrossingDecoder(
            feature_channels=2,
            feature_stride=2,
        )
        model = CURELiteFactorizedModel(adapter, decoder).eval()
        base_forward_calls = 0
        decoder_forward_calls = 0

        def count_base(*_args: object) -> None:
            nonlocal base_forward_calls
            base_forward_calls += 1

        def count_decoder(*_args: object) -> None:
            nonlocal decoder_forward_calls
            decoder_forward_calls += 1

        handles = [
            adapter.base.register_forward_hook(count_base),
            decoder.register_forward_hook(count_decoder),
        ]
        images = torch.randn(2, 1, 8, 8)
        try:
            with torch.no_grad():
                direct = model(
                    images,
                    residual_threshold=0.37,
                )
            assert adapter.extract_calls == 1
            assert base_forward_calls == 1
            assert decoder_forward_calls == 1

            with torch.no_grad():
                inferred = model.infer(
                    images,
                    residual_threshold=0.37,
                )
        finally:
            for handle in handles:
                handle.remove()

    assert adapter.extract_calls == 2
    assert adapter.base.forward_calls == 2
    assert base_forward_calls == 2
    assert decoder_forward_calls == 2
    assert tuple(direct) == tuple(inferred)
    for field_name in direct:
        assert torch.equal(direct[field_name], inferred[field_name])
    assert torch.equal(
        inferred.final_mask,
        inferred.occupancy | inferred.residual_mask,
    )
