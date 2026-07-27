from __future__ import annotations

import json
from math import log

import pytest
import torch
from torch.nn import functional as F

from cure_lite.phase_resolved_relation_training import (
    PFCR_TRAIN_NEGATIVE_PROBABILITY,
    PFCR_TRAIN_POSITIVE_PROBABILITY,
    PhaseResolvedRelationTrainingConfig,
    phase_resolved_worst_endpoint_loss,
    run_phase_resolved_relation_development,
)


def test_worst_endpoint_loss_uses_extreme_pixels_not_population_mean() -> None:
    logits = torch.tensor(
        [[[[4.0, -1.0, -4.0, 3.0]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor(
        [[[[True, True, False, False]]]],
        dtype=torch.bool,
    )
    valid = torch.ones_like(target)
    occupancy = torch.zeros_like(target)
    margin = log(
        PFCR_TRAIN_POSITIVE_PROBABILITY
        / PFCR_TRAIN_NEGATIVE_PROBABILITY
    )

    fields = phase_resolved_worst_endpoint_loss(
        logits,
        target,
        valid,
        occupancy,
        logit_margin=margin,
    )

    expected_positive = F.softplus(torch.tensor(margin - (-1.0)))
    expected_negative = F.softplus(torch.tensor(margin + 3.0))
    assert fields.positive_min_logit.item() == pytest.approx(-1.0)
    assert fields.negative_max_logit.item() == pytest.approx(3.0)
    assert fields.positive_loss.item() == pytest.approx(
        expected_positive.item()
    )
    assert fields.negative_loss.item() == pytest.approx(
        expected_negative.item()
    )
    assert fields.loss.item() == pytest.approx(
        0.5 * (expected_positive + expected_negative).item()
    )
    fields.loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 0, 1].item() != 0.0
    assert logits.grad[0, 0, 0, 3].item() != 0.0
    assert logits.grad[0, 0, 0, 0].item() == 0.0
    assert logits.grad[0, 0, 0, 2].item() == 0.0


def test_targetless_state_has_explicit_negative_only_semantics() -> None:
    logits = torch.tensor([[[[-4.0, -2.0]]]], dtype=torch.float32)
    target = torch.zeros_like(logits, dtype=torch.bool)
    valid = torch.ones_like(target)
    occupancy = torch.zeros_like(target)
    config = PhaseResolvedRelationTrainingConfig(seed=42)

    fields = phase_resolved_worst_endpoint_loss(
        logits,
        target,
        valid,
        occupancy,
        logit_margin=config.logit_margin,
    )

    assert fields.positive_state_mask.tolist() == [False]
    assert fields.negative_state_mask.tolist() == [True]
    assert fields.positive_loss.tolist() == [0.0]
    assert fields.negative_max_logit.tolist() == [-2.0]
    assert fields.loss.item() == pytest.approx(
        F.softplus(torch.tensor(config.logit_margin - 2.0)).item()
    )


def test_seed_42_development_is_deterministic_and_passes() -> None:
    config = PhaseResolvedRelationTrainingConfig(seed=42)
    first = run_phase_resolved_relation_development(config)
    second = run_phase_resolved_relation_development(config)
    first_bytes = json.dumps(
        first,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    second_bytes = json.dumps(
        second,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert first_bytes == second_bytes
    assert first["result_fingerprint"] == second["result_fingerprint"]
    assert first["decision"]["development_learnability_pass"] is True
    assert first["final_metrics"][
        "lossless_threshold_mismatch_pixel_count"
    ] == 0
    assert first["final_metrics"]["positive_probability_min"] > 0.95
    assert first["final_metrics"]["negative_probability_max"] < 0.05
    assert first["decision"]["real_dataset_model_success_claimed"] is False
