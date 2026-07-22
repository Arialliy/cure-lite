from __future__ import annotations

import pytest
import torch

from cure_lite.decoder import project_occupancy_to_feature_grid
from cure_lite.experiment.training_pipeline import (
    decoder_visible_legal_deletions,
)
from cure_lite.instances import instances_from_binary_mask
from cure_lite.intervention import enumerate_legal_deletions
from cure_lite.matching import match_components
from cure_lite.occupancy import build_occupancy
from cure_lite.toy import ToyFrozenBaseAdapter, make_two_target_scene


def test_projection_preserves_a_sub_stride_occupied_pixel() -> None:
    occupancy = torch.zeros((1, 1, 256, 256), dtype=torch.bool)
    occupancy[0, 0, 1, 1] = True

    projected = project_occupancy_to_feature_grid(occupancy, (64, 64))

    assert projected.dtype == torch.bool
    assert projected.shape == (1, 1, 64, 64)
    assert int(torch.count_nonzero(projected)) == 1
    assert not torch.any(
        project_occupancy_to_feature_grid(
            torch.zeros_like(occupancy),
            (64, 64),
        )
    )


def test_projection_rejects_upsampling_and_nonbinary_state() -> None:
    occupancy = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
    with pytest.raises(ValueError, match="may not upsample"):
        project_occupancy_to_feature_grid(occupancy, (16, 16))
    with pytest.raises(TypeError, match="must be bool"):
        project_occupancy_to_feature_grid(occupancy.to(torch.float32), (4, 4))


def test_legal_deletion_must_change_the_projected_decoder_state() -> None:
    scene = make_two_target_scene()
    output = ToyFrozenBaseAdapter()(scene.image_batch())
    occupancy, prediction = build_occupancy(output.probability)
    gt = instances_from_binary_mask(scene.gt_mask)
    matching = match_components(prediction, gt)
    legal = enumerate_legal_deletions(
        prediction,
        gt,
        matching,
        occupancy,
    )
    assert legal

    full_grid = decoder_visible_legal_deletions(
        occupancy,
        legal,
        feature_size=tuple(occupancy.shape),
    )
    one_cell = decoder_visible_legal_deletions(
        occupancy,
        legal,
        feature_size=(1, 1),
    )

    assert full_grid == legal
    assert one_cell == ()
