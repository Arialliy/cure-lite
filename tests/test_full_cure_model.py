from __future__ import annotations

import pytest
import torch

from cure_lite.cure import (
    CUREModel,
    CUREProtocol,
    CUREResidualConfig,
    CUREResidualDecoder,
    noisy_or,
)
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.provenance import BaseCheckpointSelection
from cure_lite.cure.protocol import module_state_fingerprint
from cure_lite.toy import ToyFrozenBaseAdapter


def _protocol(base: ToyFrozenBaseAdapter, config: CUREResidualConfig) -> CUREProtocol:
    manifest = SplitManifest(
        dataset="toy",
        records=(
            SplitRecord("base-fit", "D_B", "base-fit-group", "base-fit.png"),
            SplitRecord("base-select", "D_B", "base-select-group", "base-select.png"),
            SplitRecord("source", "D_R", "source-group", "source.png"),
            SplitRecord("validation", "D_V", "validation-group", "validation.png"),
            SplitRecord("test", "D_T", "test-group", "test.png"),
        ),
    )
    return CUREProtocol.from_manifest(
        manifest,
        base_fingerprint="toy-base-checkpoint",
        base_state_fingerprint=module_state_fingerprint(base),
        adapter_fingerprint=base.fingerprint,
        preprocessing_fingerprint="toy-preprocessing",
        residual_config=config,
        base_checkpoint_selection=BaseCheckpointSelection.from_manifest(
            manifest,
            fit_sample_ids=("base-fit",),
            select_sample_ids=("base-select",),
        ),
    )


def test_noisy_or_is_exact_bounded_and_monotone() -> None:
    base = torch.tensor([0.0, 0.2, 0.8, 1.0])
    residual = torch.tensor([0.7, 0.5, 0.4, 0.9])
    fused = noisy_or(base, residual)
    torch.testing.assert_close(fused, 1.0 - (1.0 - base) * (1.0 - residual))
    assert torch.all(fused >= base)
    assert torch.all(fused >= residual)
    assert torch.all((fused >= 0.0) & (fused <= 1.0))


def test_decoder_keeps_single_pixel_coverage_on_evaluation_grid() -> None:
    decoder = CUREResidualDecoder(
        CUREResidualConfig(feature_channels=3, width=8, groups=4)
    )
    feature = torch.zeros(1, 3, 2, 2)
    probability = torch.full((1, 1, 5, 5), 0.1)
    occupancy = torch.zeros_like(probability, dtype=torch.bool)
    occupancy[0, 0, 2, 2] = True
    logits = decoder(feature, probability, occupancy)
    assert logits.shape == probability.shape


def test_full_cure_masks_residual_and_preserves_base_occupancy() -> None:
    base = ToyFrozenBaseAdapter()
    config = CUREResidualConfig(
        feature_channels=base.feature_channels,
        width=8,
        groups=4,
        occupancy_threshold=0.5,
        suppression_radius=0,
        initial_residual_probability=0.1,
    )
    decoder = CUREResidualDecoder(config)
    model = CUREModel(base, decoder, _protocol(base, config))
    image = torch.zeros(1, 1, 5, 5)
    image[0, 0, 2, 2] = 1.0

    raw = model(image)
    assert raw.occupancy[0, 0, 2, 2]
    assert raw.residual_probability[0, 0, 2, 2] == 0.0
    outside = ~raw.exclusion_mask
    torch.testing.assert_close(
        raw.residual_probability[outside],
        torch.full_like(raw.residual_probability[outside], 0.1),
    )
    torch.testing.assert_close(
        raw.final_probability,
        noisy_or(raw.base_probability, raw.residual_probability),
    )



def test_full_cure_rejects_in_place_frozen_base_mutation() -> None:
    base = ToyFrozenBaseAdapter()
    config = CUREResidualConfig(feature_channels=3, width=8, groups=4)
    model = CUREModel(base, CUREResidualDecoder(config), _protocol(base, config))
    with torch.no_grad():
        next(base.parameters()).add_(0.01)
    with pytest.raises(RuntimeError, match="parameters or buffers changed"):
        model(torch.zeros(1, 1, 5, 5))


def test_full_cure_rejects_base_load_state_dict_and_full_model_targeting() -> None:
    base = ToyFrozenBaseAdapter()
    config = CUREResidualConfig(feature_channels=3, width=8, groups=4)
    model = CUREModel(base, CUREResidualDecoder(config), _protocol(base, config))
    replacement = {
        name: value.detach().clone() for name, value in base.state_dict().items()
    }
    first_name = next(iter(replacement))
    replacement[first_name].add_(0.01)
    base.load_state_dict(replacement)
    with pytest.raises(RuntimeError, match="parameters or buffers changed"):
        model(torch.zeros(1, 1, 5, 5))

    clean_base = ToyFrozenBaseAdapter()
    clean_model = CUREModel(
        clean_base,
        CUREResidualDecoder(config),
        _protocol(clean_base, config),
    )
    with pytest.raises(RuntimeError, match="may not target the frozen base"):
        clean_model.load_state_dict(clean_model.state_dict())


def test_only_full_cure_decoder_receives_gradients() -> None:
    base = ToyFrozenBaseAdapter()
    config = CUREResidualConfig(feature_channels=3, width=8, groups=4)
    decoder = CUREResidualDecoder(config)
    model = CUREModel(base, decoder, _protocol(base, config)).train()
    image = torch.rand(1, 1, 6, 6)
    output = model(image)
    output.final_probability.mean().backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert any(parameter.grad is not None for parameter in decoder.parameters())


def test_probability_conditioning_is_not_part_of_default_core() -> None:
    config = CUREResidualConfig(feature_channels=3, width=8, groups=4)
    assert config.condition_on_probability is False
    decoder = CUREResidualDecoder(config)
    # Projected feature plus occupancy only.  Probability conditioning is an
    # explicit shortcut-risk ablation, not the full-CURE default.
    assert decoder.decode[0].in_channels == config.width + 1


def test_default_decoder_is_invariant_to_probability_given_same_occupancy() -> None:
    torch.manual_seed(0)
    config = CUREResidualConfig(feature_channels=3, width=8, groups=4)
    decoder = CUREResidualDecoder(config).eval()
    feature = torch.randn(1, 3, 3, 3)
    low = torch.full((1, 1, 7, 7), 0.1)
    high = torch.full((1, 1, 7, 7), 0.4)
    occupancy = torch.zeros_like(low, dtype=torch.bool)
    torch.testing.assert_close(
        decoder(feature, low, occupancy),
        decoder(feature, high, occupancy),
    )
