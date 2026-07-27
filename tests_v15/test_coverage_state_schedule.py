from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_observability import (
    CoverageStateObservabilityDecision,
    audit_population_observability,
)
from cure_lite.coverage_state_precomputed_cache import (
    prepare_scalar_coverage_state_cache,
)
from cure_lite.coverage_state_raw_catalog import (
    CoverageStateNaturalRecord,
    make_coverage_state_raw_catalog,
)
from cure_lite.coverage_state_schedule import (
    COVERAGE_STATE_EXPOSURE_GATE_POLICY,
    COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE,
    COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION,
    COVERAGE_STATE_SOURCE_MAXIMUM_UNIFORM_MULTIPLE,
    COVERAGE_STATE_SOURCE_MINIMUM_ESS_FRACTION,
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
    coverage_state_formal_exposure_gate,
    coverage_state_schedule_exposure_report,
    materialize_coverage_state_fused_batch,
)
from cure_lite.coverage_state_sobolev import CoverageStateSobolevConfig
from tests_v15.coverage_state_test_helpers import (
    TOY_STRIDE,
    make_feature,
    make_mask,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_factual_miss,
    make_training_scalar_cache,
)


def _schedule(*, seed: int = 42):
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=seed,
            epochs=2,
            steps_per_epoch=3,
        ),
    )
    return cache, schedule


def _duplicate_actual_input_scalar_cache():
    """Return a feasible population with three target foci per actual input."""

    base = make_training_scalar_cache()
    scene = make_mask((1, 1), (3, 4), (5, 6))
    occupancy = make_mask((7, 7))
    valid = torch.ones_like(scene)
    focus_specs = (
        (1, (1, 1)),
        (2, (3, 4)),
        (3, (5, 6)),
    )
    scene_target_ids = tuple(
        f"evaluation_gt:{target_id}" for target_id, _ in focus_specs
    )
    common = {
        "sample_id": "sample-three-focus",
        "group_id": "group-three-focus",
        "state_kind": "factual_miss",
        "feature": make_feature(100),
        "occupancy": occupancy,
        "target": scene,
        "valid_mask": valid,
        "target_ids": scene_target_ids,
        "source_row_fingerprint": stable_fingerprint(
            {"duplicate_actual_input": "source"}
        ),
        "evaluation_gt_ids": tuple(
            target_id for target_id, _ in focus_specs
        ),
        "native_gt_ids": tuple(
            target_id for target_id, _ in focus_specs
        ),
    }
    duplicate_input_records = tuple(
        CoverageStateNaturalRecord(
            record_id=f"natural-three-focus-{target_id:03d}",
            loss_valid_mask=(
                valid
                & ~occupancy
                & ~(scene & ~make_mask(focus_pixel))
            ),
            focus_target_ids=(f"evaluation_gt:{target_id}",),
            lineage_record_fingerprint=stable_fingerprint(
                {
                    "duplicate_actual_input": "lineage",
                    "target_id": target_id,
                }
            ),
            **common,
        )
        for target_id, focus_pixel in focus_specs
    )
    unique_input_records = tuple(
        make_factual_miss(variant=value)
        for value in range(101, 110)
    )
    no_miss_records = tuple(
        value
        for value in base.raw_catalog.natural_records
        if value.state_kind == "factual_no_miss"
    )
    naturals = (
        *duplicate_input_records,
        *unique_input_records,
        *no_miss_records,
    )
    catalog = make_coverage_state_raw_catalog(
        dataset="toy-duplicate-actual-input",
        feature_stride=TOY_STRIDE,
        source_fingerprint=stable_fingerprint(
            {
                "duplicate_actual_input": "schedule-adversary-v1",
                "naturals": sorted(value.record_id for value in naturals),
                "pairs": sorted(
                    value.pair_id
                    for value in base.raw_catalog.pair_records
                ),
            }
        ),
        natural_records=tuple(naturals),
        pair_records=base.raw_catalog.pair_records,
    )
    receipt = audit_population_observability(catalog)
    assert (
        receipt.decision
        is CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
    )
    return prepare_scalar_coverage_state_cache(
        catalog,
        receipt,
        CoverageStateSobolevConfig(truncation_radius=TOY_STRIDE),
    )


def test_schedule_is_exactly_reproducible_and_seed_specific() -> None:
    _, first = _schedule(seed=42)
    _, replay = _schedule(seed=42)
    _, different = _schedule(seed=43)
    assert first.canonical_payload() == replay.canonical_payload()
    assert first.schedule_fingerprint == replay.schedule_fingerprint
    assert (
        first.schedule_fingerprint
        != different.schedule_fingerprint
    )
    assert len(first.selections) == 6


def test_schedule_exposure_accounting_matches_fixed_budget() -> None:
    cache, schedule = _schedule()
    exposure = schedule.exposure_counts()
    assert sum(exposure["factual_miss"].values()) == 6 * 4
    assert sum(exposure["factual_no_miss"].values()) == 6 * 4
    assert sum(exposure["clean_positive"].values()) == 6
    assert sum(exposure["component_null"].values()) == 6
    payload = schedule.canonical_payload()
    assert payload["logical_states_per_update"] == 12
    assert payload["optimizer_exposure_accounting"] == (
        "recomputed_against_current_cache_before_use"
    )
    assert "identity_null_optimizer_exposure" not in payload
    assert "diagnostic_only_optimizer_exposure" not in payload
    assert payload["objective_invariant"] is True
    report = coverage_state_schedule_exposure_report(cache, schedule)
    assert report["branches"]["factual_miss"]["record"][
        "total_exposures"
    ] == 24
    assert report["branches"]["factual_no_miss"]["record"][
        "total_exposures"
    ] == 24
    assert report["branches"]["clean_positive"]["record"][
        "total_exposures"
    ] == 6
    assert report["branches"]["component_null"]["record"][
        "total_exposures"
    ] == 6
    assert report["factual_focus_target"]["total_exposures"] == 24
    assert report["clean_added_target"]["total_exposures"] == 6
    assert report["positive_target"]["total_exposures"] == 30
    assert report["positive_target"]["formal_gate_role"] == (
        "descriptive_only"
    )
    assert report["logical_state_source"]["total_exposures"] == 72
    assert report["logical_state_source"]["formal_gate_role"] == (
        "descriptive_only"
    )
    assert report["identity_null_optimizer_exposure"] == 0
    assert report["diagnostic_only_optimizer_exposure"] == 0
    assert report["selection_audit"]["branch_exposure_totals"] == {
        "factual_miss": 24,
        "factual_no_miss": 24,
        "clean_positive": 6,
        "component_null": 6,
    }
    assert report["selection_audit"]["logical_state_evaluations"] == 72


def test_every_selection_has_unique_natural_inputs_and_distinct_pair_sources() -> None:
    cache, schedule = _schedule()
    naturals = {
        value.record.record_id: value for value in cache.natural_records
    }
    pairs = {value.record.pair_id: value for value in cache.pair_records}
    for selection in schedule.selections:
        miss_inputs = {
            naturals[value].actual_scalar_input_fingerprint
            for value in selection.factual_miss_record_ids
        }
        no_inputs = {
            naturals[value].actual_scalar_input_fingerprint
            for value in selection.factual_no_miss_record_ids
        }
        assert len(miss_inputs) == 4
        assert len(no_inputs) == 4
        assert (
            pairs[selection.clean_positive_pair_id].record.sample_id
            != pairs[selection.component_null_pair_id].record.sample_id
        )
        assert (
            pairs[selection.component_null_pair_id].optimizer_role
            == "component_null"
        )


def test_materialization_replays_exact_selection() -> None:
    cache, schedule = _schedule()
    selection = schedule.selections[4]
    batch = materialize_coverage_state_fused_batch(
        cache,
        schedule,
        epoch=1,
        step=1,
        device="cpu",
    )
    assert batch.factual_miss.record_ids == (
        selection.factual_miss_record_ids
    )
    assert batch.factual_no_miss.record_ids == (
        selection.factual_no_miss_record_ids
    )
    assert batch.pairs.pair_ids == (
        selection.clean_positive_pair_id,
        selection.component_null_pair_id,
    )
    feature, occupancy = batch.model_inputs()
    assert feature.shape[0] == occupancy.shape[0] == 12


def test_materialization_rejects_cache_schedule_mismatch() -> None:
    cache, schedule = _schedule()
    altered = replace(schedule, cache_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="schedule and scalar cache"):
        materialize_coverage_state_fused_batch(
            cache,
            altered,
            epoch=0,
            step=0,
            device="cpu",
        )


def test_formal_exposure_gate_is_pretraining_and_support_preserving() -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig.formal(seed=42),
    )
    gate = coverage_state_formal_exposure_gate(cache, schedule)
    assert gate["all_pass"] is True
    assert gate["failed_checks"] == []
    assert gate["training_authorized"] is False
    assert gate["formal_training_authorized"] is False
    assert gate["D_V_accessed"] is False
    assert gate["D_T_accessed"] is False
    assert gate["gate_policy"]["policy"] == (
        COVERAGE_STATE_EXPOSURE_GATE_POLICY
    )
    assert gate["thresholds"] == {
        "record_minimum_ess_fraction": (
            COVERAGE_STATE_RECORD_MINIMUM_ESS_FRACTION
        ),
        "record_maximum_uniform_multiple": (
            COVERAGE_STATE_RECORD_MAXIMUM_UNIFORM_MULTIPLE
        ),
        "source_minimum_ess_fraction": (
            COVERAGE_STATE_SOURCE_MINIMUM_ESS_FRACTION
        ),
        "source_maximum_uniform_multiple": (
            COVERAGE_STATE_SOURCE_MAXIMUM_UNIFORM_MULTIPLE
        ),
        "zero_exposure_count": 0,
    }
    assert not any(
        name.startswith("positive_target/")
        or name.startswith("logical_state_source/")
        for name in gate["checks"]
    )
    assert any(
        name.startswith("factual_focus_target/")
        for name in gate["checks"]
    )
    assert any(
        name.startswith("clean_added_target/")
        for name in gate["checks"]
    )

    identity = next(
        value
        for value in cache.pair_records
        if value.optimizer_role == "identity_diagnostic"
    )
    invalid_first = replace(
        schedule.selections[0],
        component_null_pair_id=identity.record.pair_id,
    )
    invalid_schedule = replace(
        schedule,
        selections=(invalid_first, *schedule.selections[1:]),
    )
    with pytest.raises(
        ValueError,
        match="component-null slot contains the wrong pair role",
    ):
        coverage_state_formal_exposure_gate(
            cache,
            invalid_schedule,
        )


def test_schedule_builder_detects_current_cache_mutation() -> None:
    cache = make_training_scalar_cache()
    cache.raw_catalog.natural_records[0].feature[0, 0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="raw catalog changed"):
        build_coverage_state_training_schedule(
            cache,
            CoverageStateScheduleConfig(
                seed=42,
                epochs=1,
                steps_per_epoch=1,
            ),
        )


def test_first_materialization_detects_post_schedule_cache_mutation() -> None:
    cache, schedule = _schedule()
    cache.raw_catalog.natural_records[0].feature[0, 0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="raw catalog changed"):
        materialize_coverage_state_fused_batch(
            cache,
            schedule,
            epoch=0,
            step=0,
            device="cpu",
        )


def test_exposure_report_detects_post_schedule_cache_mutation() -> None:
    cache, schedule = _schedule()
    cache.raw_catalog.pair_records[0].feature[0, 0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="raw catalog changed"):
        coverage_state_schedule_exposure_report(cache, schedule)


@pytest.mark.parametrize("seed", (42, 43))
def test_formal_schedule_preserves_support_with_duplicate_actual_inputs(
    seed: int,
) -> None:
    cache = _duplicate_actual_input_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig.formal(seed=seed),
    )
    gate = coverage_state_formal_exposure_gate(cache, schedule)
    assert gate["all_pass"] is True
    assert gate["failed_checks"] == []

    report = gate["report"]
    miss_record = report["branches"]["factual_miss"]["record"]
    miss_target = report["factual_focus_target"]
    assert miss_record["support_size"] == 12
    assert miss_record["zero_exposure_count"] == 0
    assert miss_target["support_size"] == 12
    assert miss_target["zero_exposure_count"] == 0
    assert (
        miss_record["maximum_count"]
        - miss_record["minimum_count"]
        <= 1
    )

    naturals = {
        value.record.record_id: value
        for value in cache.natural_records
    }
    for selection in schedule.selections:
        assert len(
            {
                naturals[record_id].actual_scalar_input_fingerprint
                for record_id in selection.factual_miss_record_ids
            }
        ) == 4
