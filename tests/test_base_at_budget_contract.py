from __future__ import annotations

import pytest
import torch

from cure_lite.calibration import (
    CalibrationSample,
    FalseAlarmBudget,
    FrozenBaseThresholdProtocol,
    select_base_threshold_at_budget,
)
from cure_lite.config import MatchConfig, OccupancyConfig
from cure_lite.stage_a import (
    BASE_RUN_IDENTITY_FIELDS,
    COMMON_FIELDS,
    _validate_common,
)


def test_base_at_budget_ignores_candidates_above_anchor_threshold() -> None:
    empty = torch.zeros(3, 3)
    sample = CalibrationSample("empty", empty, empty, empty)

    selection = select_base_threshold_at_budget(
        [sample],
        [0.9],
        OccupancyConfig(threshold=0.5),
        MatchConfig(),
        FalseAlarmBudget(pixel_fa_budget=0.0),
    )

    assert selection.feasible
    assert selection.threshold == 0.5


def test_frozen_base_at_budget_protocol_rejects_threshold_above_anchor() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        FrozenBaseThresholdProtocol(
            base_threshold=0.6,
            anchor_occupancy_config=OccupancyConfig(threshold=0.5),
        )


def test_stage_a_registry_rejects_tau_b_above_tau_o() -> None:
    common = {key: None for key in COMMON_FIELDS}
    digest_fields = (
        "manifest_fingerprint",
        "base_fingerprint",
        "base_state_fingerprint",
        "stage_a_complete_fingerprint",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
        "d_v_base_cache_index_fingerprint",
        "d_v_base_cache_index_sha256",
        "d_r_base_cache_index_fingerprint",
        "d_r_base_cache_index_sha256",
        "d_r_state_cache_index_fingerprint",
        "d_r_state_cache_index_sha256",
        "anchor_protocol_sha256",
        "state_fingerprint",
        "efficiency_static_fingerprint",
        "efficiency_receipt_fingerprint",
    )
    for field in digest_fields:
        common[field] = "a" * 64
    common["base_run_identity"] = {
        field: (
            "reference-base-v1" if field == "producer_schema" else "a" * 64
        )
        for field in BASE_RUN_IDENTITY_FIELDS
    }
    common["tau_o"] = 0.5
    common["tau_B"] = 0.6

    with pytest.raises(RuntimeError, match="tau_B must not exceed tau_o"):
        _validate_common(common)
