from __future__ import annotations

from math import log1p, sqrt

import pytest
import torch

from cure_lite.config import MISS_ALIGNMENT_POLICY, MissAlignmentConfig
from cure_lite.sampling import (
    choose_miss_aligned_legal_identity,
    miss_alignment_descriptor,
    miss_alignment_descriptors,
    positive_region_feature_rms,
    positive_region_feature_rms_many,
    quantized_miss_alignment_descriptor,
    quantized_miss_alignment_distance,
)


def test_positive_region_feature_rms_uses_area_weights_and_channels() -> None:
    feature = torch.tensor(
        [
            [
                [[3.0, 30.0], [30.0, 30.0]],
                [[4.0, 40.0], [40.0, 40.0]],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[:2, :2] = True
    expected = sqrt((3.0**2 + 4.0**2) / 2.0)
    assert positive_region_feature_rms(feature, mask) == pytest.approx(expected)
    assert miss_alignment_descriptor(feature, mask) == pytest.approx(
        log1p(expected)
    )


def test_zero_feature_has_zero_descriptor() -> None:
    feature = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    mask = torch.ones((4, 4), dtype=torch.bool)
    assert positive_region_feature_rms(feature, mask) == 0.0
    assert miss_alignment_descriptor(feature, mask) == 0.0


def test_batched_descriptors_equal_independent_target_descriptors() -> None:
    feature = torch.arange(1, 13, dtype=torch.float32).reshape(1, 3, 2, 2)
    first = torch.zeros((4, 4), dtype=torch.bool)
    first[:2, :2] = True
    second = torch.zeros((4, 4), dtype=torch.bool)
    second[2:, 2:] = True
    masks = (first, second)
    assert positive_region_feature_rms_many(feature, masks) == pytest.approx(
        tuple(positive_region_feature_rms(feature, mask) for mask in masks)
    )
    assert miss_alignment_descriptors(feature, masks) == pytest.approx(
        tuple(miss_alignment_descriptor(feature, mask) for mask in masks)
    )


def test_miss_alignment_distance_is_fixed_half_up_quantization() -> None:
    config = MissAlignmentConfig()
    assert config.policy == MISS_ALIGNMENT_POLICY
    assert quantized_miss_alignment_distance(
        0.0,
        0.0000005,
        config,
    ) == 1
    assert quantized_miss_alignment_distance(
        0.0,
        0.00000049,
        config,
    ) == 0
    assert quantized_miss_alignment_descriptor(0.0000005, config) == 1
    assert quantized_miss_alignment_descriptor(0.00000049, config) == 0


def test_nearest_legal_selection_is_global_order_invariant_and_stable() -> None:
    candidates = (
        ("sample-z", 2, 1, 0.7),
        ("sample-b", 1, 3, 0.4),
        ("sample-a", 3, 2, 0.6),
    )
    expected = ("sample-a", 3, 2, 100_000)
    assert choose_miss_aligned_legal_identity(
        0.5,
        candidates,
    ) == expected
    assert choose_miss_aligned_legal_identity(
        0.5,
        tuple(reversed(candidates)),
    ) == expected


def test_nearest_legal_selection_uses_identity_tie_break() -> None:
    candidates = (
        ("sample-b", 1, 1, 0.4),
        ("sample-a", 2, 1, 0.6),
        ("sample-a", 1, 2, 0.4),
    )
    assert choose_miss_aligned_legal_identity(
        0.5,
        candidates,
    ) == ("sample-a", 1, 2, 100_000)


def test_miss_alignment_has_no_alternate_policy_or_uniform_fallback() -> None:
    with pytest.raises(ValueError, match="fixes the miss-alignment policy"):
        MissAlignmentConfig(policy="another-policy")
    with pytest.raises(ValueError, match="non-empty"):
        choose_miss_aligned_legal_identity(0.5, ())
