from __future__ import annotations

import pytest
import torch

from cure_lite.coverage_state_batches import (
    COVERAGE_STATE_FUSED_LOGICAL_STATES,
    CoverageStateFusedBatch,
    make_coverage_state_natural_train_batch,
    make_coverage_state_pair_train_batch,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


def _fused_batch() -> CoverageStateFusedBatch:
    cache = make_training_scalar_cache()
    misses = tuple(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_miss"
    )
    no_misses = tuple(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_no_miss"
    )
    return CoverageStateFusedBatch(
        factual_miss=make_coverage_state_natural_train_batch(
            misses,
            state_kind="factual_miss",
            device="cpu",
        ),
        factual_no_miss=make_coverage_state_natural_train_batch(
            no_misses,
            state_kind="factual_no_miss",
            device="cpu",
        ),
        pairs=make_coverage_state_pair_train_batch(
            cache.clean_positive_records[0],
            cache.component_null_records[0],
            device="cpu",
        ),
    )


def test_fused_batch_has_exact_fixed_order_twelve_state_input() -> None:
    batch = _fused_batch()
    batch.validate()
    feature, occupancy = batch.model_inputs()
    assert feature.shape[0] == COVERAGE_STATE_FUSED_LOGICAL_STATES == 12
    assert occupancy.shape[0] == 12
    assert torch.equal(feature[:4], batch.factual_miss.feature)
    assert torch.equal(feature[4:8], batch.factual_no_miss.feature)
    assert torch.equal(feature[8:10], batch.pairs.feature)
    assert torch.equal(feature[10:12], batch.pairs.feature)
    assert torch.equal(occupancy[8:10], batch.pairs.occupancy_plus)
    assert torch.equal(occupancy[10:12], batch.pairs.occupancy_minus)
    assert batch.pairs.pair_kinds == (
        "clean_positive",
        "component_null",
    )


def test_fused_batch_geometry_is_fp32_bool_and_on_one_device() -> None:
    batch = _fused_batch()
    feature, occupancy = batch.model_inputs()
    assert feature.dtype == torch.float32
    assert occupancy.dtype == torch.bool
    for targets in (
        batch.factual_miss.targets,
        batch.factual_no_miss.targets,
        batch.pairs.absolute_targets_plus,
        batch.pairs.absolute_targets_minus,
    ):
        assert targets.target_field.dtype == torch.float32
        assert targets.integration_measure.dtype == torch.float32
        assert targets.field_valid_mask.dtype == torch.bool
        assert targets.loss_valid_mask.dtype == torch.bool
        assert targets.target_field.device == feature.device
    assert batch.pairs.joint_targets.target_field_plus.dtype == torch.float32
    assert batch.pairs.joint_targets.valid_mask.dtype == torch.bool


def test_natural_batch_rejects_duplicate_actual_input() -> None:
    cache = make_training_scalar_cache()
    miss = next(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_miss"
    )
    with pytest.raises(ValueError, match="repeat records or actual inputs"):
        make_coverage_state_natural_train_batch(
            (miss, miss, miss, miss),
            state_kind="factual_miss",
            device="cpu",
        )


def test_pair_batch_rejects_diagnostic_only_component() -> None:
    cache = make_training_scalar_cache()
    diagnostic = next(
        value
        for value in cache.pair_records
        if value.optimizer_role == "diagnostic_only"
    )
    with pytest.raises(ValueError, match="visible clean and component"):
        make_coverage_state_pair_train_batch(
            cache.clean_positive_records[0],
            diagnostic,
            device="cpu",
        )


def test_pair_batch_rejects_identity_null() -> None:
    cache = make_training_scalar_cache()
    identity = next(
        value
        for value in cache.pair_records
        if value.optimizer_role == "identity_diagnostic"
    )
    with pytest.raises(ValueError, match="visible clean and component"):
        make_coverage_state_pair_train_batch(
            cache.clean_positive_records[0],
            identity,
            device="cpu",
        )


def test_selection_fingerprint_is_deterministic() -> None:
    first = _fused_batch()
    second = _fused_batch()
    assert first.selection_fingerprint == second.selection_fingerprint
