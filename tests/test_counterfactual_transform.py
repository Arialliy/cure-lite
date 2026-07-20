from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import nan

import pytest
import torch

from cure_lite.counterfactual import (
    LOCAL_CONTRAST_ATTENUATION,
    TransformConfig,
    TransformDiagnostics,
    TransformSpec,
    apply_counterfactual_transform,
    build_background_ring,
    build_soft_roi,
)


def _config(**overrides) -> TransformConfig:
    values = {
        "roi_radius": 2,
        "ring_inner_radius": 2,
        "ring_outer_radius": 4,
        "strength_grid": (0.25, 0.5, 0.75),
        "minimum_ring_pixels": 1,
    }
    values.update(overrides)
    return TransformConfig(**values)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"roi_radius": True}, TypeError),
        ({"roi_radius": 0}, ValueError),
        ({"ring_inner_radius": 1}, ValueError),
        ({"ring_outer_radius": 2}, ValueError),
        ({"strength_grid": [0.5]}, TypeError),
        ({"strength_grid": ()}, ValueError),
        ({"strength_grid": (0.5, 0.5)}, ValueError),
        ({"strength_grid": (0.75, 0.25)}, ValueError),
        ({"strength_grid": (0.0, 0.5)}, ValueError),
        ({"strength_grid": (0.5, 1.0)}, ValueError),
        ({"strength_grid": (0.5, nan)}, ValueError),
        ({"minimum_ring_pixels": False}, TypeError),
    ],
)
def test_transform_config_rejects_noncanonical_values(overrides, error) -> None:
    with pytest.raises(error):
        _config(**overrides)


def test_transform_contracts_are_frozen_and_spec_uses_finite_grid() -> None:
    config = _config()
    spec = TransformSpec(config=config, strength=0.5)
    assert spec.transform_name == LOCAL_CONTRAST_ATTENUATION
    with pytest.raises(ValueError, match="strength_grid"):
        TransformSpec(config=config, strength=0.6)
    with pytest.raises(ValueError, match="transform_name"):
        TransformSpec(config=config, strength=0.5, transform_name="other")
    with pytest.raises(FrozenInstanceError):
        spec.strength = 0.25  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_abs_delta": -1.0},
        {"mean_abs_delta": -1.0},
        {"outside_roi_max_delta": -1.0},
        {"ring_pixels": 0},
        {"ring_pixels": True},
    ],
)
def test_transform_diagnostics_reject_invalid_values(kwargs) -> None:
    values = {
        "max_abs_delta": 1.0,
        "mean_abs_delta": 0.5,
        "outside_roi_max_delta": 0.0,
        "ring_pixels": 4,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        TransformDiagnostics(**values)


def test_soft_roi_and_ring_are_deterministic_disjoint_shells() -> None:
    target = torch.zeros(9, 9, dtype=torch.bool)
    target[4, 4] = True
    config = _config()
    soft = build_soft_roi(target, config)
    ring = build_background_ring(target, config)

    assert soft.dtype == torch.float32 and soft.device.type == "cpu"
    assert soft[4, 4] == 1.0
    assert soft[4, 5] == pytest.approx(2.0 / 3.0)
    assert soft[4, 6] == pytest.approx(1.0 / 3.0)
    assert soft[4, 7] == 0.0
    assert not torch.any(ring & (soft > 0))
    assert int(torch.count_nonzero(ring)) == 56
    assert torch.equal(soft, build_soft_roi(target, config))
    assert torch.equal(ring, build_background_ring(target, config))


def test_transform_moves_each_channel_toward_ring_and_preserves_outside() -> None:
    image = torch.empty(1, 2, 9, 9, dtype=torch.float32)
    image[:, 0].fill_(2.0)
    image[:, 1].fill_(-1.0)
    target = torch.zeros(9, 9, dtype=torch.bool)
    target[4, 4] = True
    image[0, 0, 4, 4] = 10.0
    image[0, 1, 4, 4] = 3.0
    config = _config()
    spec = TransformSpec(config=config, strength=0.5)

    transformed, diagnostics = apply_counterfactual_transform(image, target, spec)
    repeated, repeated_diagnostics = apply_counterfactual_transform(
        image, target, spec
    )
    hard_roi = build_soft_roi(target, config) > 0

    assert transformed.dtype == image.dtype and transformed.device == image.device
    assert transformed[0, 0, 4, 4] == pytest.approx(6.0)
    assert transformed[0, 1, 4, 4] == pytest.approx(1.0)
    assert torch.equal(transformed[0, :, ~hard_roi], image[0, :, ~hard_roi])
    assert torch.equal(transformed, repeated)
    assert diagnostics == repeated_diagnostics
    assert diagnostics.max_abs_delta == pytest.approx(4.0)
    assert diagnostics.mean_abs_delta == pytest.approx(6.0 / (2 * 9 * 9))
    assert diagnostics.outside_roi_max_delta == 0.0
    assert diagnostics.ring_pixels == 56
    assert torch.equal(image[0, :, ~target], torch.tensor([2.0, -1.0])[:, None].expand(2, 80))


def test_soft_boundary_applies_a_smaller_attenuation_than_target_core() -> None:
    image = torch.zeros(1, 1, 9, 9)
    target = torch.zeros(9, 9, dtype=torch.bool)
    target[4, 4] = True
    image[0, 0, 4, 4] = 9.0
    image[0, 0, 4, 5] = 9.0
    image[0, 0, 4, 6] = 9.0
    spec = TransformSpec(config=_config(), strength=0.75)

    transformed, _ = apply_counterfactual_transform(image, target, spec)
    core_delta = 9.0 - float(transformed[0, 0, 4, 4])
    inner_shell_delta = 9.0 - float(transformed[0, 0, 4, 5])
    outer_shell_delta = 9.0 - float(transformed[0, 0, 4, 6])
    assert core_delta > inner_shell_delta > outer_shell_delta > 0.0


@pytest.mark.parametrize(
    ("image", "target", "error"),
    [
        (torch.zeros(2, 1, 5, 5), torch.eye(5, dtype=torch.bool), ValueError),
        (
            torch.zeros(1, 1, 5, 5, dtype=torch.int64),
            torch.eye(5, dtype=torch.bool),
            TypeError,
        ),
        (torch.zeros(1, 1, 5, 5), torch.zeros(1, 5, 5, dtype=torch.bool), ValueError),
        (torch.zeros(1, 1, 5, 5), torch.zeros(5, 5), TypeError),
        (torch.zeros(1, 1, 5, 5), torch.zeros(5, 5, dtype=torch.bool), ValueError),
        (torch.zeros(1, 1, 5, 6), torch.ones(5, 5, dtype=torch.bool), ValueError),
    ],
)
def test_transform_rejects_invalid_image_or_target(image, target, error) -> None:
    with pytest.raises(error):
        apply_counterfactual_transform(
            image,
            target,
            TransformSpec(config=_config(), strength=0.5),
        )


def test_transform_rejects_a_ring_smaller_than_configured_minimum() -> None:
    image = torch.zeros(1, 1, 3, 3)
    target = torch.zeros(3, 3, dtype=torch.bool)
    target[1, 1] = True
    config = TransformConfig(
        roi_radius=1,
        ring_inner_radius=1,
        ring_outer_radius=2,
        strength_grid=(0.5,),
        minimum_ring_pixels=1,
    )
    # The target's radius-one dilation already fills this tiny image, leaving
    # no disjoint background ring.
    with pytest.raises(ValueError, match="background ring"):
        apply_counterfactual_transform(
            image, target, TransformSpec(config=config, strength=0.5)
        )


def test_other_gt_is_protected_from_roi_and_background_estimation() -> None:
    image = torch.zeros(1, 1, 15, 15)
    target = torch.zeros(15, 15, dtype=torch.bool)
    target[7, 7] = True
    protected = torch.zeros_like(target)
    protected[7, 9] = True
    image[0, 0, 7, 7] = 1.0
    image[0, 0, 7, 9] = 10.0
    config = TransformConfig(
        roi_radius=2,
        ring_inner_radius=2,
        ring_outer_radius=4,
        strength_grid=(0.5,),
    )

    transformed, _ = apply_counterfactual_transform(
        image,
        target,
        TransformSpec(config, 0.5),
        protected_mask=protected,
    )
    ring = build_background_ring(
        target,
        config,
        protected_mask=protected,
    )

    assert not ring[7, 9]
    assert transformed[0, 0, 7, 9] == image[0, 0, 7, 9]
    assert transformed[0, 0, 7, 7] == 0.5
