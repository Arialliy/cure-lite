from __future__ import annotations

import pytest
import torch

from cure_lite.config import MatchConfig
from cure_lite.instances import instances_from_binary_mask
from cure_lite.matching import match_components
from cure_lite.types import FrozenBaseOutput
from cure_lite.toy import (
    TOY_OCCUPANCY_THRESHOLD,
    ToyFrozenBaseAdapter,
    attenuate_target,
    make_factual_miss_scene,
    make_two_target_scene,
)


def _match_scene(scene, adapter: ToyFrozenBaseAdapter):
    output = adapter(scene.image_batch())
    prediction = instances_from_binary_mask(
        output.probability[0, 0] >= TOY_OCCUPANCY_THRESHOLD
    )
    gt = instances_from_binary_mask(scene.gt_mask)
    return output, gt, match_components(prediction, gt, MatchConfig())


def test_clean_scene_contains_two_separated_matched_targets() -> None:
    scene = make_two_target_scene()
    adapter = ToyFrozenBaseAdapter()

    output, gt, match = _match_scene(scene, adapter)

    assert scene.shape == (32, 32)
    assert len(gt.instances) == 2
    assert match.cardinality == 2
    assert match.unmatched_gt_ids == frozenset()
    assert output.probability.shape == (1, 1, 32, 32)
    assert output.feature.shape == (1, 3, 32, 32)


@pytest.mark.parametrize("missed_gt_id", [1, 2])
def test_local_attenuation_misses_exactly_the_selected_target(
    missed_gt_id: int,
) -> None:
    clean = make_two_target_scene()
    attenuated = attenuate_target(clean, missed_gt_id)
    adapter = ToyFrozenBaseAdapter()

    _, _, clean_match = _match_scene(clean, adapter)
    output, _, attenuated_match = _match_scene(attenuated, adapter)

    assert clean_match.cardinality == 2
    assert attenuated_match.cardinality == 1
    assert attenuated_match.unmatched_gt_ids == frozenset({missed_gt_id})
    assert attenuated_match.matched_gt_ids == frozenset({3 - missed_gt_id})

    selected = clean.target_masks[missed_gt_id - 1]
    other = clean.target_masks[2 - missed_gt_id]
    assert torch.equal(attenuated.image[0][other], clean.image[0][other])
    assert torch.equal(
        attenuated.image[0][~selected], clean.image[0][~selected]
    )
    assert torch.all(
        output.probability[0, 0][selected] < TOY_OCCUPANCY_THRESHOLD
    )


def test_factual_miss_remains_visible_in_frozen_feature() -> None:
    scene = make_factual_miss_scene(missed_gt_id=1)
    adapter = ToyFrozenBaseAdapter()

    output, _, match = _match_scene(scene, adapter)
    missed = scene.target_masks[0]
    background = ~scene.gt_mask[0]

    assert match.unmatched_gt_ids == frozenset({1})
    assert torch.all(
        output.probability[0, 0][missed] < TOY_OCCUPANCY_THRESHOLD
    )
    # Channel zero is an identity feature: the attenuated target remains a
    # deterministic nonzero signal even though the probability is subthreshold.
    assert float(output.feature[0, 0][missed].mean()) == pytest.approx(0.35)
    assert float(output.feature[0, 0][background].max()) == 0.0


def test_toy_adapter_is_frozen_detached_deterministic_and_fingerprinted() -> None:
    scene = make_two_target_scene()
    first = ToyFrozenBaseAdapter()
    second = ToyFrozenBaseAdapter()

    first.train(True)
    one = first(scene.image_batch())
    two = first(scene.image_batch())

    assert not first.training
    assert not first.base.training
    assert all(not parameter.requires_grad for parameter in first.parameters())
    assert not one.probability.requires_grad
    assert not one.feature.requires_grad
    assert one.probability.shape[-2:] == one.feature.shape[-2:]
    assert torch.equal(one.probability, two.probability)
    assert torch.equal(one.feature, two.feature)
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert set(first.fingerprint) <= set("0123456789abcdef")


def test_toy_adapter_supports_batches_and_rejects_wrong_channels() -> None:
    clean = make_two_target_scene()
    missed = make_factual_miss_scene(missed_gt_id=2)
    adapter = ToyFrozenBaseAdapter()

    batch = torch.stack((clean.image, missed.image), dim=0)
    output = adapter(batch)
    assert output.probability.shape == (2, 1, 32, 32)
    assert output.feature.shape == (2, 3, 32, 32)

    with pytest.raises(ValueError, match="single-channel"):
        adapter(torch.zeros(1, 2, 32, 32))


def test_frozen_base_contract_rejects_feature_dtype_incompatible_with_decoder() -> None:
    with pytest.raises(TypeError, match="feature must be float32"):
        FrozenBaseOutput(
            probability=torch.zeros(1, 1, 4, 4, dtype=torch.float32),
            feature=torch.zeros(1, 2, 4, 4, dtype=torch.float16),
        )
